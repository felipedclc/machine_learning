# import os
import io
import warnings
import getpass
from PIL import Image
from stability_sdk import client
from IPython.display import display
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


# key = os.environ["API_KEY"]
key = "sk-pLhjohbuTLKT0eqATl52LONYt3wCvv889aH3jOHWszjxjpoC"
print("xablau")
getpass.getpass(key)


# return a python generator
stability_api = client.StabilityInference(
    key=key,
    verbose=True,
)

answers = stability_api.generate(
    prompt="The Batman dark knight rises 4k hd wallper"
)

# print(answers)
i = 0
for resp in answers:
    i += 1
    print("resp ", i)
    for artifact in resp.artifacts:
        print("entrou no artifact")
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and cold not be proceded, Please modify the prompt and try again."
            )
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            display(img)
