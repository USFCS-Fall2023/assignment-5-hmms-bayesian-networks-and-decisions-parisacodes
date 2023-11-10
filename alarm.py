from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
    state_names={"Burglary": ['no', 'yes']},
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
    state_names={"Earthquake": ["no", "yes"]},
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
    state_names={"Burglary": ['no', 'yes'], "Earthquake": ['no', 'yes'], 'Alarm': ['yes', 'no']},
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={"Alarm": ['yes', 'no'], "JohnCalls": ['yes', 'no']},
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={"Alarm": ['yes', 'no'], "MaryCalls": ['yes', 'no']},
)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)

alarm_infer = VariableElimination(alarm_model)


def print_alarm_cpd(cpd):
    # Perform inference on the Bayesian Network
    query1_result = alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"})
    query2_result = alarm_infer.query(variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"})
    query3_result = alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls": "yes"})

    # Print the results
    print("Probability of Mary making a call given that John has called: ")
    print(query1_result)

    print("Probability of John and Mary both making calls when the alarm has gone off: ")
    print(query2_result)

    print("Probability of the alarm going off when Mary has made a call: ")
    print(query3_result)
