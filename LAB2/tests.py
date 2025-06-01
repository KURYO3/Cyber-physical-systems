import unittest
from unittest.mock import patch, MagicMock
from main import process_meter, load_all_data, connect_db, load_last_data, save_data

class TestElectricityMeter(unittest.TestCase):
    @patch('main.load_last_data')
    @patch('main.save_data')
    def test_update_existing_meter(self, mock_save_data, mock_load_last_data):
        mock_load_last_data.return_value = {"adjusted_day": 500, "adjusted_night": 300}
        bill = process_meter("1001", 600, 400)
        self.assertEqual(bill, 370.0)
        mock_save_data.assert_called_once()
        args, kwargs = mock_save_data.call_args
        self.assertEqual(args[1], 600)
        self.assertEqual(args[2], 400)
        self.assertEqual(args[3], 600)
        self.assertEqual(args[4], 400)
        self.assertEqual(args[5], 0)

    @patch('main.load_last_data')
    @patch('main.save_data')
    def test_new_meter(self, mock_save_data, mock_load_last_data):
        mock_load_last_data.return_value = None
        bill = process_meter("2002", 100, 50)
        self.assertEqual(bill, 310.0)
        mock_save_data.assert_called_once()
        args, kwargs = mock_save_data.call_args
        self.assertEqual(args[1], 100)
        self.assertEqual(args[2], 50)
        self.assertEqual(args[3], 100)
        self.assertEqual(args[4], 50)
        self.assertEqual(args[5], 0)

    @patch('builtins.input', return_value='Y')
    @patch('main.load_last_data')
    @patch('main.save_data')
    def test_lower_night_value(self, mock_save_data, mock_load_last_data, mock_input):
        mock_load_last_data.return_value = {"adjusted_day": 500, "adjusted_night": 300}
        bill = process_meter("1001", 600, 250)
        self.assertEqual(bill, 346.0)
        mock_save_data.assert_called_once()
        args, kwargs = mock_save_data.call_args
        self.assertEqual(args[1], 600)
        self.assertEqual(args[2], 300)
        self.assertEqual(args[3], 600)
        self.assertEqual(args[4], 380)
        self.assertEqual(args[5], 1)

    @patch('builtins.input', return_value='Y')
    @patch('main.load_last_data')
    @patch('main.save_data')
    def test_lower_day_value(self, mock_save_data, mock_load_last_data, mock_input):
        mock_load_last_data.return_value = {"adjusted_day": 500, "adjusted_night": 300}
        bill = process_meter("1001", 450, 400)
        self.assertEqual(bill, 370.0)
        mock_save_data.assert_called_once()
        args, kwargs = mock_save_data.call_args
        self.assertEqual(args[1], 500)
        self.assertEqual(args[2], 400)
        self.assertEqual(args[3], 600)
        self.assertEqual(args[4], 400)
        self.assertEqual(args[5], 1)

    @patch('builtins.input', side_effect=['Y', 'Y'])
    @patch('main.load_last_data')
    @patch('main.save_data')
    def test_lower_both_values(self, mock_save_data, mock_load_last_data, mock_input):
        mock_load_last_data.return_value = {"adjusted_day": 500, "adjusted_night": 300}
        bill = process_meter("1001", 450, 250)
        self.assertEqual(bill, 346.0)
        mock_save_data.assert_called_once()
        args, kwargs = mock_save_data.call_args
        self.assertEqual(args[1], 500)
        self.assertEqual(args[2], 300)
        self.assertEqual(args[3], 600)
        self.assertEqual(args[4], 380)
        self.assertEqual(args[5], 1)