# Created by Alex Pereira

# Import Libraries
import os

class CreateStorage:
    def __init__(self, actions, BASE_DIR, num_videos = 30, video_len = 30) -> None:
        """
        Constructor
        """
        # Path for exported data
        DATA_PATH = os.path.join(str(BASE_DIR))

        # Tries to make the MP_Data folder
        try:
            os.mkdir(DATA_PATH)
        except:
            pass

        # Tries to make the action folders
        for action in actions:
            try:
                action_path = os.path.join(DATA_PATH, action)
                os.mkdir(action_path)
            except:
                pass
            
            # Tries to make the frame folders
            for i in range(0, num_videos):
                try:
                    seq_path = os.path.join(action_path, str(i))
                    os.mkdir(seq_path)
                except:
                    pass