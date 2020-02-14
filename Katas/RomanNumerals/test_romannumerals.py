import unittest
from Katas.RomanNumerals import romannumerals

class MyTestCase(unittest.TestCase):
    def test_init(self):
        rn = romannumerals.RomanNumerals()

    def test_roman_numeral_of_1_is_I(self):
        self.assertEqual(romannumerals.RomanNumerals().of(1), 'I')

    def test_roman_numeral_of_2_is_II(self):
        self.assertEqual(romannumerals.RomanNumerals().of(2), 'II')

    def test_roman_numeral_of_3_is_III(self):
        self.assertEqual(romannumerals.RomanNumerals().of(3), 'III')

    def test_roman_numeral_of_4_is_IV(self):
        self.assertEqual(romannumerals.RomanNumerals().of(4), 'IV')

    def test_roman_numeral_of_5_is_V(self):
        self.assertEqual(romannumerals.RomanNumerals().of(5), 'V')

    def test_roman_numeral_of_8_is_VIII(self):
        self.assertEqual(romannumerals.RomanNumerals().of(8), 'VIII')

    def test_roman_numeral_of_9_is_IX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(9), 'IX')

    def test_roman_numeral_of_10_is_X(self):
        self.assertEqual(romannumerals.RomanNumerals().of(10), 'X')

    def test_ones_place_of_13_is_3(self):
        self.assertEqual(romannumerals.RomanNumerals()._ones_place_of(13), 3)

    def test_ones_place_of_15_is_5(self):
        self.assertEqual(romannumerals.RomanNumerals()._ones_place_of(15), 5)

    def test_roman_numeral_of_11_is_XI(self):
        self.assertEqual(romannumerals.RomanNumerals().of(11), 'XI')

    def test_roman_numeral_of_12_is_XII(self):
        self.assertEqual(romannumerals.RomanNumerals().of(12), 'XII')

    def test_roman_numeral_of_14_is_XIV(self):
        self.assertEqual(romannumerals.RomanNumerals().of(14), 'XIV')

    def test_roman_numeral_of_15_is_XV(self):
        self.assertEqual(romannumerals.RomanNumerals().of(15), 'XV')

    def test_roman_numeral_of_19_is_XIX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(19), 'XIX')

    def test_tens_place_of_39_is_3(self):
        self.assertEqual(romannumerals.RomanNumerals()._tens_place_of(39), 3)

    def test_tens_place_of_25_is_XXV(self):
        self.assertEqual(romannumerals.RomanNumerals().of(25), 'XXV')

    def test_roman_number_of_39_is_XXXIX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(39), 'XXXIX')

    def test_roman_number_of_50_is_L(self):
        self.assertEqual(romannumerals.RomanNumerals().of(50), 'L')


    def test_roman_number_of_49_is_XLIX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(49), 'XLIX')

    def test_roman_number_of_75_is_LXXV(self):
        self.assertEqual(romannumerals.RomanNumerals().of(75), 'LXXV')

    def test_roman_number_of_100_is_C(self):
        self.assertEqual(romannumerals.RomanNumerals().of(100), 'C')

    def test_roman_number_of_101_is_CI(self):
        self.assertEqual(romannumerals.RomanNumerals().of(101), 'CI')

    def test_roman_number_of_499_is_CDXCIX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(499), 'CDXCIX')

    def test_roman_number_of_500_is_D(self):
        self.assertEqual(romannumerals.RomanNumerals().of(500), 'D')

    def test_roman_number_of_666_is_DCLXVI(self):
        self.assertEqual(romannumerals.RomanNumerals().of(666), 'DCLXVI')

    def test_roman_number_of_999_is_DCLXVI(self):
        self.assertEqual(romannumerals.RomanNumerals().of(999), 'CMXCIX')

    def test_roman_number_of_1000_is_M(self):
        self.assertEqual(romannumerals.RomanNumerals().of(1000), 'M')

    def test_roman_number_of_1999_is_MCMXCIX(self):
        self.assertEqual(romannumerals.RomanNumerals().of(1999), 'MCMXCIX')



if __name__ == '__main__':
    unittest.main()
