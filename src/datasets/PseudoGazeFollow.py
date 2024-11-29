from src.datasets.GazeFollow import GazeFollow


class PseudoGazeFollow(GazeFollow):
    def add_pseudo_annotations(self, key, value):
        if key in self.pseudo_annotations:
            raise ValueError(f"Key {key} already exists")

        self.pseudo_annotations[key] = value
        self.pseudo_gaze_keys.append(key)

    def _get_gaze_coords(self, sample_key, row):
        gaze_x = self.pseudo_annotations[sample_key]["gaze_coords"][0]
        gaze_y = self.pseudo_annotations[sample_key]["gaze_coords"][1]

        return [gaze_x, gaze_y]
