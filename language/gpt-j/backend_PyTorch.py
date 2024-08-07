import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import transformers
#from ipex_llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
import mlperf_loadgen as lg
from dataset import Dataset
#import intel_extension_for_pytorch as ipex


gen_kwargs = {
    "early_stopping": False,
    "max_new_tokens": 128,
    "min_new_tokens": 1,
    "do_sample": False,
    "num_beams": 1, # only beam_size 4 is allowed for official submission
}


class SUT_base():
    def __init__(self, model, dtype, dataset_path, max_examples, use_gpu=False):
        # TODO : Pass model file name to init instead of args
        print("Loading PyTorch model...")
        self.dataset_path = dataset_path
        self.model_path = "/mnt/disk1/llm-models/Llama-2-7b-chat-hf"
        self.model_name = self.model_path
        self.use_gpu = use_gpu
        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            print("BF16 autocast")
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32


        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_object = Dataset(
            self.dataset_path, total_count_override=max_examples)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

            # Cast to GPU
            if self.use_gpu:
                input_ids_tensor = input_ids_tensor.to("xpu")
                input_masks_tensor = input_masks_tensor.to("xpu")

            pred_output_batch = self.inference_call(
                input_ids_tensor, input_masks_tensor).cpu().numpy()

            n_tokens = pred_output_batch[0].shape[0]
            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()

            response = [lg.QuerySampleResponse(
                query_samples[i].id, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)


    def inference_call(self, input_ids_tensor, input_masks_tensor):
        ''' Common for all scenarios '''
        torch_device_type = 'cuda' if self.use_gpu else 'cpu'

        with torch.inference_mode():
            input_batch = dict()
            input_batch['input_ids'] = input_ids_tensor
            input_batch['attention_mask'] = input_masks_tensor

            output_batch = self.model.generate(
                **input_batch, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)

            output_batch = output_batch.cpu()

            input_batch_lengths = [x.shape[0]
                                   for x in input_batch["input_ids"]]

            output_batch_lengths = [x.shape[0] for x in output_batch]

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)

            output_batch_str = self.tokenizer.batch_decode(output_batch_truncated, skip_special_tokens=False)


        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model, dtype, dataset_path, max_examples, use_gpu):
        SUT_base.__init__(self, model, dtype, dataset_path, max_examples, use_gpu)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu):

        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to("xpu")
            input_masks_tensor = input_masks_tensor.to("xpu")

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()
    
        n_tokens = pred_output_batch[0].shape[0]

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1], n_tokens)]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(self, model, dtype, dataset_path, max_examples, use_gpu):
        SUT_base.__init__(self, model, dtype, dataset_path, max_examples, use_gpu)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to("xpu")
            input_masks_tensor = input_masks_tensor.to("xpu")

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()
        
        n_tokens = pred_output_batch.shape[0]
        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1], n_tokens)]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(model, scenario, dtype, dataset_path, max_examples, use_gpu=False):
    if scenario == "Offline":
        return SUT_Offline(model, dtype, dataset_path, max_examples, use_gpu)
    elif scenario == "Server":
        return SUT_Server(model, dtype, dataset_path, max_examples, use_gpu)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model, dtype, dataset_path, max_examples, use_gpu)
