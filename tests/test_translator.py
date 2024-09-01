import unittest
from src.translator import EnglishKoreanTranslator

class TestEnglishKoreanTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = EnglishKoreanTranslator()

    def test_translation(self):
        result = self.translator.translate("Hello, world!")
        self.assertEqual(result, "안녕, 세상!")  # Example expected output

if __name__ == '__main__':
    unittest.main()
