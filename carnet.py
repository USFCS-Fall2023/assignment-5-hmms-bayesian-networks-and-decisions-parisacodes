from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("KeyPresent", "Starts"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ['Works', "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ['Full', "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works', "Doesn't work"]}
)

# cpd_starts = TabularCPD(
#     variable="Starts",
#     variable_card=2,
#     values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
#     evidence=["Ignition", "Gas"],
#     evidence_card=[2, 2],
#     state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
# )

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01],  # P(starts | gas, ignition, keyPresent)
        [0.99, 0.01, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99],  # P(starts | gas, ignition, !keyPresent)
    ],
    evidence=["Gas", "Ignition", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Gas": ['Full', 'Empty'],
        "Ignition": ["Works", "Doesn't work"],
        "KeyPresent": ['yes', 'no'],
    },
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no']}
)

cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent": ['yes', 'no']},
)

# Associating the parameters with the model structure
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)

car_infer = VariableElimination(car_model)


# print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))


def print_carnet_cpd(cpd):
    # Perform inference on the Bayesian Network
    query1_result = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    query2_result = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    query3_result_1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    query3_result_2 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Empty"})
    query4_result_1 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    query4_result_2 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Full"})
    query5_result = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})

    # Print the results
    print("Probability that the battery is not working when the car doesn't move: ")
    print(query1_result)

    print("Probability that the car will not start when the radio doesn't turn on: ")
    print(query2_result)

    print("Probability of the radio working when the battery is working and the car has gas (Full): ")
    print(query3_result_1)

    print("Probability of the radio working when the battery is working and the car has no gas (Empty): ")
    print(query3_result_2)

    print("Probability of the ignition failing when the car doesn't move and it has no gas (Empty): ")
    print(query4_result_1)

    print("Probability of the ignition failing when the car doesn't move and it has gas (Full): ")
    print(query4_result_2)

    print("Probability that the car starts when the radio works and the car has gas (Full): ")
    print(query5_result)
