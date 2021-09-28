# encoding: utf-8
# module compressai.ans
# from D:\ProgramData\Anaconda3\envs\python37\lib\site-packages\compressai-1.1.1-py3.7-win-amd64.egg\compressai\ans.cp37-win_amd64.pyd
# by generator 1.147
""" range Asymmetric Numeral System python bindings """

# imports
import pybind11_builtins as __pybind11_builtins


# no functions
# classes

class BufferedRansEncoder(__pybind11_builtins.pybind11_object):
    # no doc
    def encode_with_indexes(self, arg0, p_int=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ encode_with_indexes(self: compressai.ans.BufferedRansEncoder, arg0: List[int], arg1: List[int], arg2: List[List[int]], arg3: List[int], arg4: List[int]) -> None """
        pass

    def flush(self): # real signature unknown; restored from __doc__
        """ flush(self: compressai.ans.BufferedRansEncoder) -> bytes """
        return b""

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: compressai.ans.BufferedRansEncoder) -> None """
        pass


class RansDecoder(__pybind11_builtins.pybind11_object):
    # no doc
    def decode_stream(self, arg0, p_int=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ decode_stream(self: compressai.ans.RansDecoder, arg0: List[int], arg1: List[List[int]], arg2: List[int], arg3: List[int]) -> List[int] """
        pass

    def decode_with_indexes(self, arg0, arg1, p_int=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """
        decode_with_indexes(self: compressai.ans.RansDecoder, arg0: str, arg1: List[int], arg2: List[List[int]], arg3: List[int], arg4: List[int]) -> List[int]
        
        Decode a string to a list of symbols
        """
        pass

    def set_stream(self, arg0): # real signature unknown; restored from __doc__
        """ set_stream(self: compressai.ans.RansDecoder, arg0: str) -> None """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: compressai.ans.RansDecoder) -> None """
        pass


class RansEncoder(__pybind11_builtins.pybind11_object):
    # no doc
    def encode_with_indexes(self, arg0, p_int=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ encode_with_indexes(self: compressai.ans.RansEncoder, arg0: List[int], arg1: List[int], arg2: List[List[int]], arg3: List[int], arg4: List[int]) -> bytes """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: compressai.ans.RansEncoder) -> None """
        pass


# variables with complex values

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x00000292B3E8C4C8>'

__spec__ = None # (!) real value is "ModuleSpec(name='compressai.ans', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x00000292B3E8C4C8>, origin='D:\\\\ProgramData\\\\Anaconda3\\\\envs\\\\python37\\\\lib\\\\site-packages\\\\compressai-1.1.1-py3.7-win-amd64.egg\\\\compressai\\\\ans.cp37-win_amd64.pyd')"

