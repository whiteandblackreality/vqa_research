from pathlib import Path

from src.loader import VQA


class VQADatasetLoader:
    def __init__(self,
                 version: str = '2.0',
                 path_to_data_dir: Path = Path('../../data'),
                 task_type: str = 'OpenEnded',
                 data_type: str = 'mscoco',
                 data_sub_type: str = 'train2014'):

        self._version = version
        self._path_to_data_dir = path_to_data_dir
        self._task_type = task_type
        self._data_type = data_type
        self._data_sub_type = data_sub_type

        self.annotation_file = Path(path_to_data_dir,
                                    'annotations',
                                    f'{version}_{data_type}_{data_sub_type}_annotations.json')

        self.question_file = Path(path_to_data_dir,
                                  'questions',
                                  f'{version}_{task_type}_{data_type}_{data_sub_type}_questions.json')

        self.image_dir = Path(path_to_data_dir,
                              'images',
                              data_type,
                              data_sub_type)

        self.vqa_api = VQA(annotation_file=self.annotation_file,
                           question_file=self.question_file)

    def __repr__(self):
        return 'Class for VQA dataset loading'
