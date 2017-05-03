# lumped-hydrological
Lumped hydrological models for sreamflow simulation and forecasting

This module contains the implementation of 3 lumped hydrological models: HBV-96, Sugawara tank and Linear Tank model.

The libraries contain 2 fundamental functions, which are `simulate` and `calibrate`. The first is used to calculate discharge, based on the required data for each of the models. The latter is used to define the optimal set of parameters, based on a metric of error between the simulated and measured discharge.

Please, if you happen to find any errors in the code, or just would like to further collaborate in the development of these codes, feel free to drop a line in a message.

Regards,
Juan Chacon-Hurtado
