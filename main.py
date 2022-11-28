from entrance import EntranceState
from exit import ExitState


def main():
    try:
        entrance = EntranceState(dev=True)
        entrance.start()
        exit = ExitState(dev=True)
        exit.start()
        while entrance.is_running() and exit.is_running():
            pass
    except:
        entrance.stop()
        exit.stop()


if __name__ == '__main__':
    main()
