from .infer.worldmodel import Worldinfer
from PIL import Image

class RWKVVisualGen:
    def __init__(self, 
        llm_path: str, 
        encoder_path: str, 
        encoder_type: str='siglip'):
        
        # prepare the model
        self.model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

    def generate(self, img: Image.Image, text: str) -> str:
        result, _ = self.model.generate(text, img)  # type: ignore
        return result
