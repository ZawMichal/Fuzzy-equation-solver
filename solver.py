from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import numpy as np
import os
import platform
import subprocess
from pathlib import Path
from typing import Callable

from models import FuzzyNumber


class SolveCancelled(RuntimeError):
	"""Sygnalizuje kooperatywne przerwanie obliczeń."""


class FuzzySystemSolver:
	"""
	Każdy rozmyty współczynnik parametryzowany jest zmienną RDM ∈ [0, 1]
	na bazie horyzontalnych funkcji przynależności. Obwiednie (span)
	wyznaczane są hybrydowo po wymiarach: dla `d` wybranych parametrów
	stosowane jest RDM (wierzchołki), a dla pozostałych metoda HMF.
	"""

	DEFAULT_VERTEX_BATCH_SIZE = 8192
	MIN_VERTEX_BATCH_SIZE = 256
	MAX_VERTEX_BATCH_SIZE = 16384
	MAX_VERTEX_BATCH_SIZE_LARGE_L3 = 32768
	LARGE_L3_THRESHOLD_BYTES = 96 * 1024 * 1024
	DEFAULT_TARGET_BATCH_BYTES = 8 * 1024 * 1024
	DEFAULT_FALLBACK_LEAF_SIZE = 1024
	MIN_FALLBACK_LEAF_SIZE = 64
	DEFAULT_AUTO_ALPHA_STEPS = 21
	GAUSSIAN_AUTO_ALPHA_STEPS = 31

	_cache_profile: dict[str, int] | None = None

	def __init__(self, vertex_limit: int = 18) -> None:
		self.vertex_limit = max(0, int(vertex_limit))
		self._cache_info = self._get_cache_profile()

	def _auto_mu_levels(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
	) -> np.ndarray:
		"""Buduje automatyczne poziomy μ bez ręcznego parametru m.

		Dla układów z gaussami używa nieco gęstszej siatki, aby stabilniej
		odwzorować szerokie ogony nośnika przy małych wartościach μ.
		"""
		has_gaussian = any(cell.kind == "gaussian" for row in a_matrix for cell in row)
		has_gaussian = has_gaussian or any(cell.kind == "gaussian" for cell in b_vector)

		steps = self.GAUSSIAN_AUTO_ALPHA_STEPS if has_gaussian else self.DEFAULT_AUTO_ALPHA_STEPS
		raw = np.linspace(0.0, 1.0, steps, dtype=float)
		if has_gaussian:
			# Delikatne zagęszczenie przy małych μ (szersze przekroje gaussowskie).
			raw = raw * raw
		levels = np.unique(np.clip(raw, 0.0, 1.0))
		if levels[0] > 0.0:
			levels = np.insert(levels, 0, 0.0)
		if levels[-1] < 1.0:
			levels = np.append(levels, 1.0)
		return levels

	@staticmethod
	def _parse_cache_size_to_bytes(raw: str) -> int:
		text = raw.strip().upper()
		if not text:
			return 0

		multiplier = 1
		if text.endswith("KB"):
			multiplier = 1024
			text = text[:-2]
		elif text.endswith("MB"):
			multiplier = 1024 * 1024
			text = text[:-2]
		elif text.endswith("GB"):
			multiplier = 1024 * 1024 * 1024
			text = text[:-2]
		elif text.endswith("K"):
			multiplier = 1024
			text = text[:-1]
		elif text.endswith("M"):
			multiplier = 1024 * 1024
			text = text[:-1]
		elif text.endswith("G"):
			multiplier = 1024 * 1024 * 1024
			text = text[:-1]

		try:
			return int(float(text) * multiplier)
		except ValueError:
			return 0

	@classmethod
	def _detect_windows_cache_bytes(cls) -> tuple[int, int]:
		l2_values_kb: list[int] = []
		l3_values_kb: list[int] = []

		# Prefer WMIC when present (fast and simple), fallback to PowerShell CIM.
		try:
			proc = subprocess.run(
				["wmic", "cpu", "get", "L2CacheSize,L3CacheSize", "/value"],
				check=False,
				capture_output=True,
				text=True,
				timeout=1.5,
			)
			if proc.returncode == 0 and proc.stdout:
				for line in proc.stdout.splitlines():
					if "=" not in line:
						continue
					key, value = line.split("=", 1)
					key = key.strip().upper()
					value = value.strip()
					if not value.isdigit():
						continue
					cache_kb = int(value)
					if key == "L2CACHESIZE" and cache_kb > 0:
						l2_values_kb.append(cache_kb)
					elif key == "L3CACHESIZE" and cache_kb > 0:
						l3_values_kb.append(cache_kb)
		except (OSError, subprocess.SubprocessError):
			pass

		if not l2_values_kb and not l3_values_kb:
			try:
				ps_script = (
					"Get-CimInstance Win32_Processor | "
					"Select-Object -Property L2CacheSize,L3CacheSize | "
					"ConvertTo-Json -Compress"
				)
				proc = subprocess.run(
					["powershell", "-NoProfile", "-Command", ps_script],
					check=False,
					capture_output=True,
					text=True,
					timeout=2.0,
				)
				if proc.returncode == 0 and proc.stdout:
					import json

					parsed = json.loads(proc.stdout)
					rows = parsed if isinstance(parsed, list) else [parsed]
					for row in rows:
						if not isinstance(row, dict):
							continue
						l2_val = int(row.get("L2CacheSize") or 0)
						l3_val = int(row.get("L3CacheSize") or 0)
						if l2_val > 0:
							l2_values_kb.append(l2_val)
						if l3_val > 0:
							l3_values_kb.append(l3_val)
			except (OSError, ValueError, subprocess.SubprocessError):
				pass

		l2_bytes = max(l2_values_kb) * 1024 if l2_values_kb else 0
		l3_bytes = max(l3_values_kb) * 1024 if l3_values_kb else 0
		return l2_bytes, l3_bytes

	@classmethod
	def _detect_linux_cache_bytes(cls) -> tuple[int, int]:
		cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
		if not cache_root.exists():
			return 0, 0

		l2_bytes = 0
		l3_bytes = 0
		for index_path in cache_root.glob("index*"):
			try:
				level = (index_path / "level").read_text(encoding="utf-8").strip()
				cache_type = (index_path / "type").read_text(encoding="utf-8").strip().lower()
				size_text = (index_path / "size").read_text(encoding="utf-8").strip()
			except OSError:
				continue

			if cache_type not in ("data", "unified"):
				continue

			size_bytes = cls._parse_cache_size_to_bytes(size_text)
			if level == "2":
				l2_bytes = max(l2_bytes, size_bytes)
			elif level == "3":
				l3_bytes = max(l3_bytes, size_bytes)

		return l2_bytes, l3_bytes

	@classmethod
	def _detect_macos_cache_bytes(cls) -> tuple[int, int]:
		def _sysctl_int(key: str) -> int:
			try:
				proc = subprocess.run(
					["sysctl", "-n", key],
					check=False,
					capture_output=True,
					text=True,
					timeout=1.0,
				)
				if proc.returncode != 0:
					return 0
				value = proc.stdout.strip()
				return int(value) if value.isdigit() else 0
			except (OSError, ValueError, subprocess.SubprocessError):
				return 0

		return _sysctl_int("hw.l2cachesize"), _sysctl_int("hw.l3cachesize")

	@classmethod
	def _get_cache_profile(cls) -> dict[str, int]:
		if cls._cache_profile is not None:
			return cls._cache_profile

		l2_bytes = 0
		l3_bytes = 0
		system = platform.system().lower()

		if system == "windows":
			l2_bytes, l3_bytes = cls._detect_windows_cache_bytes()
		elif system == "linux":
			l2_bytes, l3_bytes = cls._detect_linux_cache_bytes()
		elif system == "darwin":
			l2_bytes, l3_bytes = cls._detect_macos_cache_bytes()

		target_batch_bytes = cls.DEFAULT_TARGET_BATCH_BYTES
		if l3_bytes > 0:
			target_batch_bytes = max(cls.DEFAULT_TARGET_BATCH_BYTES, int(0.35 * l3_bytes))
		elif l2_bytes > 0:
			target_batch_bytes = max(cls.DEFAULT_TARGET_BATCH_BYTES // 2, int(0.50 * l2_bytes))

		cls._cache_profile = {
			"l2_bytes": int(max(0, l2_bytes)),
			"l3_bytes": int(max(0, l3_bytes)),
			"target_batch_bytes": int(max(1, target_batch_bytes)),
		}
		return cls._cache_profile

	def _vertex_batch_size(self, param_dim: int, n: int, num_vertices: int) -> int:
		max_batch_size = self._max_vertex_batch_size()

		override = os.getenv("FUZZY_SOLVER_BATCH_SIZE", "").strip()
		if override:
			try:
				override_value = int(override)
				clamped_override = max(
					self.MIN_VERTEX_BATCH_SIZE,
					min(max_batch_size, override_value),
				)
				return min(num_vertices, clamped_override)
			except ValueError:
				pass

		target_batch_bytes = int(self._cache_info.get("target_batch_bytes", self.DEFAULT_TARGET_BATCH_BYTES))
		# RDM + vals to największa część pamięci per wierzchołek; A/b są widokami z vals.
		bytes_per_vertex = max(1, 16 * int(param_dim) + 8 * int(n))
		estimated = target_batch_bytes // bytes_per_vertex

		if estimated <= 0:
			estimated = self.MIN_VERTEX_BATCH_SIZE

		clamped = max(self.MIN_VERTEX_BATCH_SIZE, min(max_batch_size, int(estimated)))
		return min(num_vertices, clamped)

	def _max_vertex_batch_size(self) -> int:
		l3_bytes = int(self._cache_info.get("l3_bytes", 0))
		if l3_bytes >= self.LARGE_L3_THRESHOLD_BYTES:
			return self.MAX_VERTEX_BATCH_SIZE_LARGE_L3
		return self.MAX_VERTEX_BATCH_SIZE

	def cache_profile(self) -> dict[str, int]:
		"""Zwraca profil cache używany do auto-doboru batch size (L2/L3 + cel pamięciowy)."""
		return {
			"l2_bytes": int(self._cache_info.get("l2_bytes", 0)),
			"l3_bytes": int(self._cache_info.get("l3_bytes", 0)),
			"target_batch_bytes": int(self._cache_info.get("target_batch_bytes", self.DEFAULT_TARGET_BATCH_BYTES)),
		}

	@staticmethod
	def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
		if cancel_check is not None and cancel_check():
			raise SolveCancelled()

	def _vertex_parallel_workers(self, num_batches: int) -> int:
		if num_batches <= 1:
			return 1

		min_batches_override = os.getenv("FUZZY_SOLVER_MIN_PARALLEL_BATCHES", "").strip()
		try:
			min_parallel_batches = int(min_batches_override) if min_batches_override else 4
		except ValueError:
			min_parallel_batches = 4
		if num_batches < max(2, min_parallel_batches):
			return 1

		workers_override = os.getenv("FUZZY_SOLVER_VERTEX_THREADS", "").strip()
		if workers_override:
			try:
				workers = max(1, int(workers_override))
				return min(num_batches, workers)
			except ValueError:
				pass

		cpu_total = os.cpu_count() or 1
		reserve_override = os.getenv("FUZZY_SOLVER_THREAD_RESERVE", "").strip()
		if reserve_override:
			try:
				reserve = max(0, int(reserve_override))
			except ValueError:
				reserve = min(4, max(1, cpu_total // 4))
		else:
			reserve = min(4, max(1, cpu_total // 4))

		usable = max(1, cpu_total - reserve)
		return min(num_batches, usable)

	def _fallback_leaf_size(self, batch_count: int) -> int:
		override = os.getenv("FUZZY_SOLVER_FALLBACK_LEAF", "").strip()
		if override:
			try:
				value = int(override)
				return max(self.MIN_FALLBACK_LEAF_SIZE, min(int(batch_count), value))
			except ValueError:
				pass
		return max(self.MIN_FALLBACK_LEAF_SIZE, min(int(batch_count), self.DEFAULT_FALLBACK_LEAF_SIZE))

	def _solve_vertex_batch(
		self,
		lo: np.ndarray,
		widths: np.ndarray,
		fuzzy_idx: np.ndarray,
		n: int,
		batch_start: int,
		batch_end: int,
		cancel_check: Callable[[], bool] | None = None,
	) -> tuple[np.ndarray, np.ndarray, bool]:
		batch_count = batch_end - batch_start
		x_lo = np.full(n, np.inf)
		x_hi = np.full(n, -np.inf)

		# Wektoryzowane generowanie bitów RDM redukuje narzut pętli interpretowanych.
		if len(fuzzy_idx) <= 63:
			vertex_ids = np.arange(batch_start, batch_end, dtype=np.uint64)
			vals_batch = np.broadcast_to(lo, (batch_count, len(lo))).copy()
			if len(fuzzy_idx) > 0:
				bit_positions = np.arange(len(fuzzy_idx), dtype=np.uint64)
				rdm_bits = ((vertex_ids[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(float)
				vals_batch[:, fuzzy_idx] = lo[fuzzy_idx] + rdm_bits * widths[fuzzy_idx]
			self._check_cancel(cancel_check)
		else:
			rdm_batch = np.zeros((batch_count, len(lo)))
			for v_idx in range(batch_count):
				if (v_idx & 0x1FF) == 0:
					self._check_cancel(cancel_check)
				bits = batch_start + v_idx
				for k_pos, fi in enumerate(fuzzy_idx):
					rdm_batch[v_idx, fi] = 1.0 if (bits >> k_pos) & 1 else 0.0
			self._check_cancel(cancel_check)
			vals_batch = lo + widths * rdm_batch
		A_batch = vals_batch[:, : n * n].reshape(batch_count, n, n)
		b_batch = vals_batch[:, n * n :]
		leaf_size = self._fallback_leaf_size(batch_count)

		def _solve_recursive(
			A_curr: np.ndarray,
			b_curr: np.ndarray,
			depth: int,
		) -> tuple[np.ndarray, np.ndarray, bool]:
			curr_count = int(A_curr.shape[0])
			local_lo = np.full(n, np.inf)
			local_hi = np.full(n, -np.inf)

			try:
				x_batch_local = np.linalg.solve(
					A_curr, b_curr[:, :, np.newaxis],
				).squeeze(-1)
				valid_local = np.all(np.isfinite(x_batch_local), axis=1)
				if np.any(valid_local):
					xv = x_batch_local[valid_local]
					local_lo = np.minimum(local_lo, np.min(xv, axis=0))
					local_hi = np.maximum(local_hi, np.max(xv, axis=0))
				has_local = not np.any(np.isinf(local_lo))
				return local_lo, local_hi, has_local
			except np.linalg.LinAlgError:
				pass

			if curr_count <= leaf_size:
				for v in range(curr_count):
					if (v & 0xFF) == 0:
						self._check_cancel(cancel_check)
					try:
						x = np.linalg.solve(A_curr[v], b_curr[v])
					except np.linalg.LinAlgError:
						x, residuals, rank, _ = np.linalg.lstsq(A_curr[v], b_curr[v], rcond=None)
						if rank < n:
							continue
						if residuals.size and not np.all(np.isfinite(residuals)):
							continue
					if np.all(np.isfinite(x)):
						local_lo = np.minimum(local_lo, x)
						local_hi = np.maximum(local_hi, x)
				has_local = not np.any(np.isinf(local_lo))
				return local_lo, local_hi, has_local

			mid = curr_count // 2
			if mid <= 0 or mid >= curr_count:
				has_local = False
				return local_lo, local_hi, has_local

			left_lo, left_hi, left_has = _solve_recursive(A_curr[:mid], b_curr[:mid], depth + 1)
			right_lo, right_hi, right_has = _solve_recursive(A_curr[mid:], b_curr[mid:], depth + 1)

			if left_has:
				local_lo = np.minimum(local_lo, left_lo)
				local_hi = np.maximum(local_hi, left_hi)
			if right_has:
				local_lo = np.minimum(local_lo, right_lo)
				local_hi = np.maximum(local_hi, right_hi)

			has_local = not np.any(np.isinf(local_lo))
			return local_lo, local_hi, has_local

		try:
			x_batch = np.linalg.solve(
				A_batch, b_batch[:, :, np.newaxis],
			).squeeze(-1)
			valid = np.all(np.isfinite(x_batch), axis=1)
			if np.any(valid):
				xv = x_batch[valid]
				x_lo = np.minimum(x_lo, np.min(xv, axis=0))
				x_hi = np.maximum(x_hi, np.max(xv, axis=0))
		except np.linalg.LinAlgError:
			# W przypadku osobliwości dzielimy batch i schodzimy rekurencyjnie
			# tylko do problematycznych fragmentów.
			x_lo, x_hi, _has_result_recursive = _solve_recursive(A_batch, b_batch, 0)

		has_result = not np.any(np.isinf(x_lo))
		return x_lo, x_hi, has_result

	@staticmethod
	def _parametrize_rdm(
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		mu: float,
		n: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Zwraca (lo, widths, fuzzy_idx) – parametryzację RDM alfa-przekrojów."""
		d = n * n + n
		lo = np.empty(d)
		widths = np.empty(d)

		for i in range(n):
			for j in range(n):
				left, right = a_matrix[i][j].alpha_cut(mu)
				k = i * n + j
				lo[k] = left
				widths[k] = right - left
		for i in range(n):
			left, right = b_vector[i].alpha_cut(mu)
			k = n * n + i
			lo[k] = left
			widths[k] = right - left

		fuzzy_idx = np.where(np.abs(widths) > 1e-15)[0]
		return lo, widths, fuzzy_idx

	@staticmethod
	def _build_system(
		lo: np.ndarray,
		widths: np.ndarray,
		rdm: np.ndarray,
		n: int,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Buduje crisp A i b dla wektora RDM: val = lo + rdm · widths."""
		vals = lo + widths * rdm
		return vals[: n * n].reshape(n, n), vals[n * n :]

	def _vertex_solve(
		self,
		lo: np.ndarray,
		widths: np.ndarray,
		fuzzy_idx: np.ndarray,
		n: int,
		progress_tick: Callable[[int], None] | None = None,
		cancel_check: Callable[[], bool] | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Dokładne obwiednie – pełne wyliczenie wierzchołków {0,1}^d."""
		self._check_cancel(cancel_check)
		d_fuzzy = len(fuzzy_idx)
		if d_fuzzy == 0:
			# Wszystkie parametry są dokładne, zatem istnieje jedno rozwiązanie punktowe.
			rdm = np.zeros_like(lo)
			A, b_vec = self._build_system(lo, widths, rdm, n)
			try:
				x = np.linalg.solve(A, b_vec)
			except np.linalg.LinAlgError as exc:
				raise ValueError("Układ osobliwy") from exc
			if progress_tick is not None:
				progress_tick(1)
			return x.copy(), x.copy()

		x_lo = np.full(n, np.inf)
		x_hi = np.full(n, -np.inf)

		num_vertices = 1 << d_fuzzy
		batch_size = self._vertex_batch_size(param_dim=len(lo), n=n, num_vertices=num_vertices)
		batch_ranges: list[tuple[int, int]] = []
		for batch_start in range(0, num_vertices, batch_size):
			batch_end = min(batch_start + batch_size, num_vertices)
			batch_ranges.append((batch_start, batch_end))

		workers = self._vertex_parallel_workers(num_batches=len(batch_ranges))
		if workers <= 1:
			for batch_start, batch_end in batch_ranges:
				self._check_cancel(cancel_check)
				batch_lo, batch_hi, has_result = self._solve_vertex_batch(
					lo, widths, fuzzy_idx, n, batch_start, batch_end, cancel_check,
				)
				if has_result:
					x_lo = np.minimum(x_lo, batch_lo)
					x_hi = np.maximum(x_hi, batch_hi)
				if progress_tick is not None:
					progress_tick(1)
		else:
			pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vertex-batch")
			futures = [
				pool.submit(
					self._solve_vertex_batch,
					lo,
					widths,
					fuzzy_idx,
					n,
					batch_start,
					batch_end,
					cancel_check,
				)
				for batch_start, batch_end in batch_ranges
			]
			pending = set(futures)
			try:
				while pending:
					self._check_cancel(cancel_check)
					done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
					for future in done:
						batch_lo, batch_hi, has_result = future.result()
						if has_result:
							x_lo = np.minimum(x_lo, batch_lo)
							x_hi = np.maximum(x_hi, batch_hi)
						if progress_tick is not None:
							progress_tick(1)
			except SolveCancelled:
				for future in pending:
					future.cancel()
				raise
			finally:
				# Czekamy na zakończenie uruchomionych zadań, aby nie zostawiać
				# aktywnych wątków po wyjściu z solve (stabilność UI/aplikacji).
				pool.shutdown(wait=True, cancel_futures=True)

		if np.any(np.isinf(x_lo)):
			raise ValueError("Układ osobliwy")
		return x_lo, x_hi

	@staticmethod
	def _mul_interval(a_l: float, a_u: float, b_l: float, b_u: float) -> tuple[float, float]:
		"""Mnoży dwa przedziały [a_l, a_u] · [b_l, b_u] metodą czterech iloczynów."""
		p1 = a_l * b_l
		p2 = a_l * b_u
		p3 = a_u * b_l
		p4 = a_u * b_u
		return min(p1, p2, p3, p4), max(p1, p2, p3, p4)

	@staticmethod
	def _div_interval(n_l: float, n_u: float, d_l: float, d_u: float) -> tuple[float, float]:
		"""Dzieli przedział [n_l, n_u] przez [d_l, d_u] w arytmetyce przedziałowej.

		Gdy mianownik zawiera zero, zwracany jest przedział nieograniczony (−∞, +∞).
		"""
		if d_l <= 0.0 <= d_u:
			return -np.inf, np.inf
		r1 = 1.0 / d_l
		r2 = 1.0 / d_u
		r_l = min(r1, r2)
		r_u = max(r1, r2)
		return FuzzySystemSolver._mul_interval(n_l, n_u, r_l, r_u)

	def _hmf_solve_from_bounds(
		self,
		vals_l: np.ndarray,
		vals_u: np.ndarray,
		n: int,
		progress_tick: Callable[[int], None] | None = None,
		cancel_check: Callable[[], bool] | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Szybka obwiednia HMF: przybliżenie środkiem + krótka relaksacja przedziałowa."""
		A_l = vals_l[: n * n].reshape(n, n)
		A_u = vals_u[: n * n].reshape(n, n)
		b_l = vals_l[n * n :]
		b_u = vals_u[n * n :]

		A_mid = 0.5 * (A_l + A_u)
		b_mid = 0.5 * (b_l + b_u)
		try:
			x_mid = np.linalg.solve(A_mid, b_mid)
		except np.linalg.LinAlgError:
			x_mid, _residuals, rank, _ = np.linalg.lstsq(A_mid, b_mid, rcond=None)
			if int(rank) < int(n):
				raise ValueError("Układ osobliwy")

		try:
			inv_mid = np.linalg.inv(A_mid)
		except np.linalg.LinAlgError:
			inv_mid = np.linalg.pinv(A_mid)

		A_rad = 0.5 * np.abs(A_u - A_l)
		b_rad = 0.5 * np.abs(b_u - b_l)
		radius = np.abs(inv_mid) @ (A_rad @ np.abs(x_mid) + b_rad)
		x_l = x_mid - radius
		x_u = x_mid + radius

		# Dwie szybkie iteracje relaksacji znacząco zwężają przedziały przy małym koszcie.
		for _ in range(2):
			self._check_cancel(cancel_check)
			changed = False
			for i in range(n):
				s_l = 0.0
				s_u = 0.0
				for j in range(n):
					if j == i:
						continue
					t_l, t_u = self._mul_interval(A_l[i, j], A_u[i, j], x_l[j], x_u[j])
					s_l += t_l
					s_u += t_u

				n_l = b_l[i] - s_u
				n_u = b_u[i] - s_l
				c_l, c_u = self._div_interval(n_l, n_u, A_l[i, i], A_u[i, i])
				if np.isfinite(c_l) and np.isfinite(c_u):
					new_l = max(x_l[i], c_l)
					new_u = min(x_u[i], c_u)
					if new_l <= new_u:
						if abs(new_l - x_l[i]) > 1e-12 or abs(new_u - x_u[i]) > 1e-12:
							changed = True
						x_l[i], x_u[i] = new_l, new_u
			if not changed:
				break

		if progress_tick is not None:
			progress_tick(1)

		if np.any(~np.isfinite(x_l)) or np.any(~np.isfinite(x_u)):
			raise ValueError("Układ osobliwy")
		if np.any(x_l > x_u):
			raise ValueError("Układ osobliwy")
		return x_l, x_u

	def _hybrid_param_solve(
		self,
		lo: np.ndarray,
		widths: np.ndarray,
		fuzzy_idx: np.ndarray,
		n: int,
		d_limit: int,
		progress_tick: Callable[[int], None] | None = None,
		cancel_check: Callable[[], bool] | None = None,
	) -> tuple[np.ndarray, np.ndarray, int]:
		"""Hybryda wymiarowa: RDM dla d_limit zmiennych, HMF dla pozostałych."""
		self._check_cancel(cancel_check)
		d_fuzzy = len(fuzzy_idx)
		if d_fuzzy == 0:
			rdm = np.zeros_like(lo)
			A, b_vec = self._build_system(lo, widths, rdm, n)
			try:
				x = np.linalg.solve(A, b_vec)
			except np.linalg.LinAlgError as exc:
				raise ValueError("Układ osobliwy") from exc
			if progress_tick is not None:
				progress_tick(1)
			return x.copy(), x.copy(), 0

		sampled_dims = min(max(0, int(d_limit)), d_fuzzy)
		if sampled_dims == 0:
			vals_l = lo.copy()
			vals_u = lo + widths
			xl, xu = self._hmf_solve_from_bounds(vals_l, vals_u, n, progress_tick, cancel_check)
			return xl, xu, 0

		order = np.argsort(np.abs(widths[fuzzy_idx]))[::-1]
		selected = fuzzy_idx[order[:sampled_dims]]
		selected_idx = selected.astype(int, copy=False)
		selected_lo = lo[selected_idx]
		selected_widths = widths[selected_idx]
		bit_positions = np.arange(sampled_dims, dtype=np.int64)

		x_lo = np.full(n, np.inf)
		x_hi = np.full(n, -np.inf)
		num_vertices = 1 << sampled_dims

		base_l = lo.copy()
		base_u = lo + widths

		for bits in range(num_vertices):
			self._check_cancel(cancel_check)
			vals_l = base_l.copy()
			vals_u = base_u.copy()
			bit_mask = (np.right_shift(bits, bit_positions) & 1).astype(float)
			selected_vals = selected_lo + selected_widths * bit_mask
			vals_l[selected_idx] = selected_vals
			vals_u[selected_idx] = selected_vals

			try:
				xl_v, xu_v = self._hmf_solve_from_bounds(vals_l, vals_u, n, None, cancel_check)
				if np.all(np.isfinite(xl_v)) and np.all(np.isfinite(xu_v)):
					x_lo = np.minimum(x_lo, xl_v)
					x_hi = np.maximum(x_hi, xu_v)
			except ValueError:
				continue

			if progress_tick is not None:
				progress_tick(1)

		if np.any(np.isinf(x_lo)) or np.any(np.isinf(x_hi)):
			raise ValueError("Układ osobliwy")
		return x_lo, x_hi, sampled_dims

	def solve(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		progress_callback: Callable[[int, int], None] | None = None,
		cancel_check: Callable[[], bool] | None = None,
	) -> dict[str, np.ndarray]:
		"""Rozwiązuje układ rozmyty i zwraca obwiednie rozwiązań po poziomach μ.

		Args:
			a_matrix: Macierz współczynników jako liczby rozmyte.
			b_vector: Wektor prawej strony jako liczby rozmyte.

		Returns:
			Słownik z kluczami ``alpha``, ``x_lower``, ``x_upper``.
		"""
		n = len(a_matrix)
		self._check_cancel(cancel_check)
		mu_levels = self._auto_mu_levels(a_matrix, b_vector)
		level_count = len(mu_levels)
		x_lower = np.zeros((level_count, n), dtype=float)
		x_upper = np.zeros((level_count, n), dtype=float)
		d_fuzzy_levels = np.zeros(level_count, dtype=int)
		method_flags = np.zeros(level_count, dtype=np.int8)
		sampled_vertices_levels = np.zeros(level_count, dtype=int)
		sampled_dims_levels = np.zeros(level_count, dtype=int)
		vertex_levels = 0
		hmf_levels = 0
		hybrid_levels = 0

		level_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]] = []
		d_limit = max(0, int(self.vertex_limit))
		for mu in mu_levels:
			lo, widths, fuzzy_idx = self._parametrize_rdm(a_matrix, b_vector, float(mu), n)
			d_fuzzy = len(fuzzy_idx)
			batch_size = self._vertex_batch_size(param_dim=len(lo), n=n, num_vertices=max(1, (1 << d_fuzzy) if d_fuzzy > 0 else 1))
			if d_limit == 0:
				mode_flag = 1  # pełny HMF
				units = 1
				effective_dims = 0
			elif d_limit >= d_fuzzy:
				mode_flag = 0  # pełny RDM
				num_vertices = 1 if d_fuzzy == 0 else (1 << d_fuzzy)
				units = max(1, (num_vertices + batch_size - 1) // batch_size)
				effective_dims = d_fuzzy
			else:
				mode_flag = 2  # hybryda wymiarowa
				num_vertices = 1 << d_limit
				units = num_vertices
				effective_dims = d_limit
			level_data.append((lo, widths, fuzzy_idx, mode_flag, units, effective_dims))

		total_units = 0
		for _lo, _w, _fuzzy_idx, _mode_flag, units, _effective_dims in level_data:
			total_units += units

		done_units = 0
		# Raportowanie postępu z gęstszą i liniową dyskretyzacją progów.
		max_reports = min(20000, max(400, int(total_units)))
		threshold_ratios = np.linspace(0.0, 1.0, max_reports + 1)
		threshold_units = np.unique(
			np.clip(np.ceil(threshold_ratios * total_units).astype(int), 0, total_units)
		)
		next_threshold_idx = 0

		def progress_tick(increment: int) -> None:
			nonlocal done_units, next_threshold_idx
			done_units = min(total_units, done_units + increment)
			if progress_callback is None:
				return
			while (
				next_threshold_idx < len(threshold_units)
				and done_units >= int(threshold_units[next_threshold_idx])
			):
				progress_callback(done_units, total_units)
				next_threshold_idx += 1

		if progress_callback is not None:
			progress_callback(0, max(1, total_units))

		for mu_idx, (lo, widths, fuzzy_idx, mode_flag, _units, effective_dims) in enumerate(level_data):
			self._check_cancel(cancel_check)
			d_fuzzy = len(fuzzy_idx)
			d_fuzzy_levels[mu_idx] = d_fuzzy

			if mode_flag == 0:
				xl, xu = self._vertex_solve(lo, widths, fuzzy_idx, n, progress_tick, cancel_check)
				vertex_levels += 1
			elif mode_flag == 1:
				vals_l = lo.copy()
				vals_u = lo + widths
				xl, xu = self._hmf_solve_from_bounds(vals_l, vals_u, n, progress_tick, cancel_check)
				method_flags[mu_idx] = 1
				sampled_vertices_levels[mu_idx] = 0
				sampled_dims_levels[mu_idx] = 0
				hmf_levels += 1
			else:
				xl, xu, sampled_dims = self._hybrid_param_solve(
					lo, widths, fuzzy_idx, n, effective_dims, progress_tick, cancel_check,
				)
				method_flags[mu_idx] = 2
				sampled_vertices_levels[mu_idx] = 1 << sampled_dims
				sampled_dims_levels[mu_idx] = sampled_dims
				hybrid_levels += 1

			x_lower[mu_idx, :] = xl
			x_upper[mu_idx, :] = xu

		# Zapewnienie monotoniczności obwiedni względem poziomu μ.
		for mu_idx in range(1, level_count):
			x_lower[mu_idx, :] = np.maximum(
				x_lower[mu_idx, :], x_lower[mu_idx - 1, :]
			)
			x_upper[mu_idx, :] = np.minimum(
				x_upper[mu_idx, :], x_upper[mu_idx - 1, :]
			)

		# Zapewnienie relacji porządkowej lower <= upper.
		for mu_idx in range(level_count):
			mid = (x_lower[mu_idx, :] + x_upper[mu_idx, :]) / 2.0
			x_lower[mu_idx, :] = np.minimum(x_lower[mu_idx, :], mid)
			x_upper[mu_idx, :] = np.maximum(x_upper[mu_idx, :], mid)

		if progress_callback is not None:
			progress_callback(max(1, total_units), max(1, total_units))

		return {
			"alpha": mu_levels,
			"x_lower": x_lower,
			"x_upper": x_upper,
			"meta": {
				"alpha_mode": "auto-horizontal",
				"alpha_levels_count": level_count,
				"d_fuzzy_levels": d_fuzzy_levels,
				"method_flags": method_flags,
				"sampled_vertices_levels": sampled_vertices_levels,
				"sampled_dims_levels": sampled_dims_levels,
				"vertex_levels": vertex_levels,
				"hmf_levels": hmf_levels,
				"hybrid_levels": hybrid_levels,
				"monotonicity_levels": hmf_levels,
				"sampled_vertices_total": int(np.sum(sampled_vertices_levels)),
				"sampled_dims_max": int(np.max(sampled_dims_levels)),
				"vertex_limit": self.vertex_limit,
				"rdm_dims_requested": d_limit,
				"cache_l2_bytes": int(self._cache_info.get("l2_bytes", 0)),
				"cache_l3_bytes": int(self._cache_info.get("l3_bytes", 0)),
				"cache_target_batch_bytes": int(self._cache_info.get("target_batch_bytes", self.DEFAULT_TARGET_BATCH_BYTES)),
			},
		}
