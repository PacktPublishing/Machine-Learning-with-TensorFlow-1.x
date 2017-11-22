# Example of Saving Model

This is an example of how to save the GraphDef and variables of a converted model 
in the Tensorflow official form. By doing this, the converted model can be 
conveniently applied with Tensorflow APIs in other languages.

For example, if a converted model is named "VGG", the generated code file should
be named as "VGG.py", and the class name inside should remain "CaffeNet".

The module "VGG" should be able to be directly imported. So put it inside the
[save_graphdef](save_graphdef) folder, or add it to "sys.path".

To save model variables, pass the path of the converted data file (e.g. VGG.npy)
to the parameter "--data-input-path".

A "VGG_frozen.pb' is also generated with all variables converted into constants
in the saved graph.