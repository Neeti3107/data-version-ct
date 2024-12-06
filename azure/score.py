import os
import logging
import json
import numpy as np
import joblib
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "carmodel.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")


pandas_sample_input = PandasParameterType(pd.DataFrame({ 'KM_Driven':[10],'Fuel_Type':['Petrol'],'age':[4],'Transmission':['Automatic'],'Owner_Type':['First'],'Seats':[4],'make':['maruti'],'mileage_new':[21],'engine_new':[900],'model':['swift'],'power_new':[90],'Location':['Bangalore']}))
output_sample = np.array([5.2])

@input_schema('data', pandas_sample_input)
# 'Inputs' is case sensitive
@output_schema(NumpyParameterType(output_sample))

def run(data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()
