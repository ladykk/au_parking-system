from modules.entrance import EntranceState
from modules.exit import ExitState


def main():
    entrance = EntranceState()
    entrance.start()
    exit = ExitState()
    exit.start()

    while entrance.is_running() and exit.is_running():
        pass


if __name__ == '__main__':
    main()
