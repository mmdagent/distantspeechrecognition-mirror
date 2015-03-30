# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.
import _test
def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class Foo_Impl(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Foo_Impl, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Foo_Impl, name)
    def __init__(self,*args,**kwargs):
        _swig_setattr(self, Foo_Impl, 'this', apply(_test.new_Foo_Impl,args, kwargs))
        _swig_setattr(self, Foo_Impl, 'thisown', 1)
    __swig_setmethods__["x"] = _test.Foo_Impl_x_set
    __swig_getmethods__["x"] = _test.Foo_Impl_x_get
    if _newclass:x = property(_test.Foo_Impl_x_get,_test.Foo_Impl_x_set)
    def bar(*args, **kwargs): return apply(_test.Foo_Impl_bar,args, kwargs)
    def __del__(self, destroy= _test.delete_Foo_Impl):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __repr__(self):
        return "<C Foo_Impl instance at %s>" % (self.this,)

class Foo_ImplPtr(Foo_Impl):
    def __init__(self,this):
        _swig_setattr(self, Foo_Impl, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Foo_Impl, 'thisown', 0)
        _swig_setattr(self, Foo_Impl,self.__class__,Foo_Impl)
_test.Foo_Impl_swigregister(Foo_ImplPtr)

class FooPtr(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FooPtr, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FooPtr, name)
    def __init__(self,*args,**kwargs):
        _swig_setattr(self, FooPtr, 'this', apply(_test.new_FooPtr,args, kwargs))
        _swig_setattr(self, FooPtr, 'thisown', 1)
    def __deref__(*args, **kwargs): return apply(_test.FooPtr___deref__,args, kwargs)
    def __del__(self, destroy= _test.delete_FooPtr):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __repr__(self):
        return "<C FooPtr instance at %s>" % (self.this,)

class FooPtrPtr(FooPtr):
    def __init__(self,this):
        _swig_setattr(self, FooPtr, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, FooPtr, 'thisown', 0)
        _swig_setattr(self, FooPtr,self.__class__,FooPtr)
_test.FooPtr_swigregister(FooPtrPtr)

make_Foo = _test.make_Foo

do_something = _test.do_something


