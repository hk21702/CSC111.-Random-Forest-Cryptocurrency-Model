from configuration import Config
import initialization

if __name__ == '__main__':
    initialization.create_project_dirs()
    config = Config()
    initialization.run()