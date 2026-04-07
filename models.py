import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FuzzyNumber:
	"""Reprezentuje liczbę rozmytą o zadanym typie funkcji przynależności.

	Atrybut `kind` określa rodzinę funkcji (np. trójkątna, trapezowa, Gaussa),
	natomiast `params` przechowuje jej parametry geometryczne.
	"""

	kind: str
	params: tuple

	def membership(self, x: float) -> float:
		"""Zwraca wartość funkcji przynależności mu(x) dla punktu `x`.

		Args:
			x: Argument, dla którego wyznaczana jest przynależność.

		Returns:
			Wartość z przedziału [0, 1] opisująca stopień przynależności.
		"""

		kind = self.kind
		p = self.params
		if kind == "crisp":
			return 1.0 if x == p[0] else 0.0
		if kind == "triangular":
			l, m, r = p
			if x < l or x > r:
				return 0.0
			if m == l and x <= m:
				return 1.0
			if m == r and x >= m:
				return 1.0
			if x <= m:
				return (x - l) / (m - l)
			return (r - x) / (r - m)
		if kind == "trapezoid":
			l, ml, mr, r = p
			if x < l or x > r:
				return 0.0
			if ml <= x <= mr:
				return 1.0
			if x < ml:
				return 1.0 if ml == l else (x - l) / (ml - l)
			return 1.0 if mr == r else (r - x) / (r - mr)
		if kind == "rectangle":
			l, r = p
			return 1.0 if l <= x <= r else 0.0
		if kind == "gaussian":
			mean, sigma = p
			return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
		raise ValueError("Nieznany typ funkcji przynależności")

	def alpha_cut(self, alpha: float) -> tuple[float, float]:
		"""Oblicza przekrój alfa liczby rozmytej.

		Dla rozkładu Gaussa granice przekroju wyznaczane są jako:
		mean minus/plus sigma razy pierwiastek z minus dwa razy logarytm alfa
		dla alpha większego od zera.
		Dla alpha mniejszego lub równego zero przyjmowane jest praktyczne
		ograniczenie nośnika do przedziału mean minus/plus 3 * sigma.

		Args:
			alpha: Poziom przekroju alfa z przedziału [0, 1].

		Returns:
			Krotka `(lewa_granica, prawa_granica)` przekroju.
		"""

		kind = self.kind
		p = self.params
		if kind == "crisp":
			v = p[0]
			return v, v
		if kind == "triangular":
			l, m, r = p
			return l + alpha * (m - l), r - alpha * (r - m)
		if kind == "trapezoid":
			l, ml, mr, r = p
			return l + alpha * (ml - l), r - alpha * (r - mr)
		if kind == "rectangle":
			l, r = p
			return l, r
		if kind == "gaussian":
			mean, sigma = p
			if alpha <= 0:
				d = 3.0 * sigma
			else:
				d = sigma * math.sqrt(-2.0 * math.log(alpha))
			return mean - d, mean + d
		raise ValueError("Nieznany typ funkcji przynależności")

	def representative(self) -> float:
		"""Wyznacza wartość reprezentatywną liczby rozmytej.

		Wartość ta jest używana m.in. do wyznaczania trendów i estymacji
		punktów przecięcia w warstwie wizualizacji.

		Returns:
			Skalar reprezentujący liczbę rozmytą.
		"""

		kind = self.kind
		p = self.params
		if kind == "crisp":
			return p[0]
		if kind == "triangular":
			return p[1]
		if kind == "trapezoid":
			return (p[1] + p[2]) / 2.0
		if kind == "rectangle":
			return (p[0] + p[1]) / 2.0
		if kind == "gaussian":
			return p[0]
		raise ValueError("Nieznany typ funkcji przynależności")
