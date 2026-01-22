import torch
import torch.nn as nn

import timm
import numpy

import os

from transformers import BitsAndBytesConfig, PretrainedConfig, PreTrainedModel, BertModel, AutoConfig, AutoModel, BertConfig
from peft import get_peft_model,IA3Config #, LoraConfig

from models.classifiers.graph_model import model_graph
from models.classifiers.mlp_model import model_mlp
from models.classifiers.mean_agre_model import model_mean_aggre

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.makedirs('/workspace/results/.cache', exist_ok=True)

os.environ["HF_HOME"] = "/workspace/results/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/results/.cache"
os.environ["HF_MODULES_CACHE"] = "/workspace/results/.cache"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available(), "  n_gpus:", torch.cuda.device_count())

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class KEEPConfig(PretrainedConfig):
    model_type = "keep"  # 

    def __init__(
        self,
        vision_config=None,  # Vision Encoder
        text_config=None,    # Text Encoder
        projection_dim=768,  
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.projection_dim = projection_dim
        

class KEEPModel(PreTrainedModel):
    config_class = KEEPConfig  # 

    def __init__(self, config):
        super().__init__(config)

        # Vision Encoder 
        self.visual = timm.create_model("vit_large_patch16_224",
                                        pretrained=False,
                                        img_size=224,
                                        patch_size=16,
                                        init_values=1e-5,
                                        num_classes=0,
                                        dynamic_img_size=True,
        )

        self.visual_head = nn.Sequential(
                    nn.Linear(self.visual.num_features, config.projection_dim),
                    nn.GELU(),
                    nn.Linear(config.projection_dim, config.projection_dim)
                )

        # Text Encoder
        text_config =  BertConfig(**config.text_config)
        self.text = BertModel(text_config)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * numpy.log(1 / 0.04))

    def encode_image(self, image_inputs):
        vision_features = self.visual(image_inputs)  # [batch_size, vision_dim]
        vision_features =  torch.nn.functional.normalize(self.visual_head(vision_features), dim=-1)  # [batch_size, projection_dim]
        
        return vision_features
    
    def encode_text(self, text_inputs):
        text_features = torch.nn.functional.normalize(self.text(**text_inputs).pooler_output, dim=-1)  # [batch_size, text_dim]
        return text_features
    
    
    def forward(self, image_inputs, text_inputs):
        vision_features = self.encode_image(image_inputs)
        
        #text_features = self.encode_text(text_inputs)

        return {
            "vision_features": vision_features,  
            #"text_features": text_features       
        }

def get_keep_ia3_model(clasification_model = 'graph',pretrain=False):

    checkpoint_path= '/workspace/data/MODELS/model_keep/pytorch_model.bin'
    model_name = 'keep'
    model_path = '/workspace/data/MODELS/model_keep/'

    AutoConfig.register(model_name, KEEPConfig)
    AutoModel.register(KEEPConfig, KEEPModel)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Modelo cuantificado en 4 bits 
        bnb_4bit_compute_dtype=torch.float16,   # Float16 para maentener la velocidad y precision 
        bnb_4bit_use_double_quant=True,         # Usa doble cuanticicacion
        bnb_4bit_quant_type="nf4",              # Variante de tipo de cuqntificacion de 4 bits 
    )
    
    config = AutoConfig.from_pretrained(model_path + 'config.json',
                                    trust_remote_code=True,             # Ejecuta el codio en remoto
                                    quantization_config=bnb_config,     # Introduce la configuracion de carga del modelo
                                    #force_download=True,                # Descarga del modelo 
                                    low_cpu_mem_usage = True            # Reduce memoria en la carga
    )
    model = AutoModel.from_config(config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
        
    # Activar gradient checkpointing para reducir VRAM
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    
    model.config.use_cache = False  # Reducir uso de memoria durante forward

    for param in model.parameters():
        param.requires_grad = False

    # Identificar módulos lineales donde aplicar IA3
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["attn.qkv", "attn.proj"]#, "mlp.fc1", "mlp.fc2"]
        ):
            target_modules.append(name)
    

    ia3_config = IA3Config(target_modules=target_modules,
        feedforward_modules=target_modules,
        task_type="FEATURE_EXTRACTION" #SEQ_CLS
    )

    ia3_model = get_peft_model(model, ia3_config)
   
    model = torch.compile(model) # Activar torch.compile si usas PyTorch 2.0+
    
    for name, param in ia3_model.named_parameters():
        if pretrain == False:
            param.requires_grad = False
    #ia3_model.print_trainable_parameters()

    # Envolver con tu modelo final
    if clasification_model == 'graph':
        model_class = model_graph()

    elif clasification_model == 'mlp':
        model_class = model_mlp(mil_aggregation_mode='attentionMIL',chunk_size=512)

    elif clasification_model == 'mean_aggre':
        model_class = model_mlp(mil_aggregation_mode='mean',chunk_size=512)
        
    elif clasification_model == 'max_aggre':
        model_class = model_mlp(mil_aggregation_mode='max',chunk_size=512)
    else:
        print("Error: clasification_model no reconocido")

    net = KEEPForGleason(ia3_model, model_class)

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Parámetros entrenables:", trainable_params)
    
    return net

class KEEPForGleason(nn.Module):
    def __init__(self, base_model, classifier_model):
        super().__init__()
        self.base = base_model
        self.classifier = classifier_model
        self.classifier.apply(weights_init)

    def forward(self, images):
        # Codificación del encoder visual
        lengths = [int(t.size(0)) for t in images]
        all_patches = torch.cat([t for t in images if t.size(0) > 0], dim=0)
        
        chunks = torch.split(all_patches, 256, dim=0)
        
        all_features =  []

        # 4. Procesar chunk a chunk en GPU
        for chunk in chunks:
            features = self.base.encode_image(chunk)
            all_features.append(features)
        total_feats = torch.cat(all_features, dim=0)
        features_list = list(torch.split(total_feats, lengths, dim=0))
        
        wsi_prediction, patchs_prediction, weights_tensor_list = self.classifier(features_list)
        return wsi_prediction, patchs_prediction, weights_tensor_list
