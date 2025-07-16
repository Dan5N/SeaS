# show_attn_utils.py is copied from the Attend-and-Excite:
# https://github.com/yuval-alaluf/Attend-and-Excite/utils/ptp_utils.py
# This is used to visualize the attention maps.

import abc
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, List

class DummyController:
    def __call__(self, *args):
        return args[0]

    def __init__(self):
        self.num_att_layers = 0

def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02) -> Image.Image:
    """ 
    Displays a list of images in a grid. 
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    return pil_img

def register_attention_control(model, controller,is_unet = False):
    """
    This function replaces the cross-attention modules in the model (Unet),
    and saves the attention maps to the controller.
    """
    def ca_forward(self, place_in_unet):
        """
        This function replaces the cross-attention modules in the model.
        It returns the function 'forward', in which the attention maps are saved in the controller
        """
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, return_cross_map=False, timestep=None, box=None, **cross_attention_kwargs):
            """
            This function is uesed to replace the forward function of CrossAttention in the model.
            """
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x) #[16,64,1280] -> [128,64,160]
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = reshape_heads_to_batch_dim(self,q)
            k = reshape_heads_to_batch_dim(self,k)
            v = reshape_heads_to_batch_dim(self,v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale #[batch * h , 64 , 77]
            cross_map = sim

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # save the attention maps
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn , v)
            out = reshape_batch_dim_to_heads(self,out)
            
            if return_cross_map is True:
                return to_out(out), cross_map
            else:
                return to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet, module_name=None):
        """
        This function counts the number of all cross-attention modules in the model.
        """
        if module_name in ["attn2"]:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for k,net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet, module_name=k)
        return count
    
    if controller is None:
        controller = DummyController()

    cross_att_count = 0
    if is_unet == False:
        sub_nets = model.unet.named_children()
    else:
        sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        if "mid" in net[0]:
           cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # save all the attention maps in step_store of one step during inference
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        # during the steps, reset the counter and step_store
        # and save the attention maps to the global_store if needed
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn    

class AttentionStore(AttentionControl):
    """
    This function replaces the cross-attention modules in the model,
    and saves the attention maps in self.attention_store.
    """
    def __init__(self, save_global_store=False):
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        This function saves the cross-attention map in the self.step_store.
        """
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        batch: int,
                        from_where: List[str],
                        is_cross: bool,
                        ) -> torch.Tensor:
    """ 
    Aggregates the attention across the different layers and heads at the specified resolution. 
    This function returns the attention maps.
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps, _  = item.chunk(2)
                cross_maps = cross_maps.reshape(batch, -1, res, res, item.shape[-1]) #[batch, head, res, res , 77]
                out.append(cross_maps)
    out = torch.stack(out, dim=0).mean(dim=0)
    out = out.mean(dim=1)
    return out
