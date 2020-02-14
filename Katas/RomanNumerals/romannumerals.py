class RomanNumerals:

    def __init__(self):
        self._roman_numerals_units = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX' ]
        self._roman_numerals_tens = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
        self._roman_numerals_hundreds = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
        self._roman_numerals_thousands = ['', 'M', 'MM', 'MMM']

    def of(self, number : int):
        s = self._roman_numerals_thousands[self._thousands_place_of(number)] + \
            self._roman_numerals_hundreds[self._hundreds_place_of(number)] + \
            self._roman_numerals_tens[self._tens_place_of(number)] + \
            self._roman_numerals_units[self._ones_place_of(number)]

        return s

    def _ones_place_of(self, number):
        return number % 10

    def _tens_place_of(self, number):
        return int(number/10)%10

    def _hundreds_place_of(self, number):
        return int(number/100)%10

    def _thousands_place_of(self, number):
        return int(number / 1000) % 10

