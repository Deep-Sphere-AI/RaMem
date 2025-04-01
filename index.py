from RaMem.RaMem import RaMemIntegratedModel

ramem = RaMemIntegratedModel()
# Generar respuesta utilizando el modelo
prompt = "quiero buscar los sintomas de la gripe"
response = ramem.generate(prompt)
print(response)