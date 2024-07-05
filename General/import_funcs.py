import os


def drive_letter(test_drives=('D', 'E')):
    for letter in test_drives:
        if os.path.exists(f'{letter}:'):
            return letter
    raise FileNotFoundError(f'No drives found, tested {', '.join(test_drives[:-1])}, and {test_drives[-1]}')
