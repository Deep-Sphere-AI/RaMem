from RaMem.RaMem import RaMemIntegratedModel

ramem = RaMemIntegratedModel()
# Generar respuesta utilizando el modelo
prompt = "podrias decirme los sintomas de la gripe?"
response = ramem.generate(prompt)
print(response)