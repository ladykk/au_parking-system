from entrance import EntranceState
from exit import ExitState


def main():
    entrance = EntranceState(dev=True)
    entrance.start()
    exit = ExitState(dev=True)
    exit.start()

    while entrance.is_running() and exit.is_running():
        pass


if __name__ == '__main__':
    main()
