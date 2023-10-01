#import required library
import random
import pandas as pd
from typing import Dict
from faker import Faker

#create a Faker instance and set the locale to GB(Great Britain)
fake = Faker(['en_GB'])

'''
generate synthetic data for the Customer table,

customer_id:- a unique identifier for each customer in the table
customer_name:- customer's name
age:- customer's age
income:- customer's yearly income
credit_score:- customer's credit score based on their credit history
debt_to_income_ratio:- the ratio of the customer's total debt to their income 
employment_status:- customer's employement status
loan_amount:- the loan amount the customer is applying for
loan_term:- specific duration of time for repayment of loan
payment_history:- customer's past payment behaviour on loans and credit accounts
number_of_dependents:- the number of people financially dependent on the customer.
'''
def generate_customer_data(num_records: int):
    customer_data: Dict[str, list] = {
        'customer_id': [fake.aba() for i in range(num_records)],
        'customer_name': [fake.name() for name in range(num_records)],
        'age': [random.randint(18, 70) for age in range(num_records)],
        'income': [random.randint(20000, 100000) for income in range(num_records)],
        'credit_score': [random.randint(300, 850) for score in range(num_records)],
        'debt_to_income_ratio': [round(random.uniform(0.1, 1.0), 2) for ratio in range(num_records)],
        'employment_status': [random.choice(['Employed', 'Unemployed', 'Self-employed']) for status in range(num_records)],
        'loan_amount': [random.randint(1000, 50000) for amount in range(num_records)],
        'loan_term': [random.choice([12, 24, 36, 48, 60]) for term in range(num_records)],
        'payment_history': [random.choice(['Good', 'Fair', 'Poor']) for history in range(num_records)],
        'number_of_dependents': [random.randint(0, 5) for dep in range(num_records)]
    }
    return customer_data

#total number of records to generate
number_of_rows: int = 900000

#generate synthetic data for the Customer table
customer_data: Dict[str, list] = generate_customer_data(number_of_rows)

#create a pandas DataFrame from the dictionary 'customer_data'
df_customer: pd.DataFrame = pd.DataFrame(customer_data)

#export the synthetic data to a CSV file
outout_file: str = 'synthetic_customer_data.csv'
df_customer.to_csv(outout_file, index=False, encoding='utf-8', header="true")

print(f"Synthetic data is created according to specific requirements and saved to {outout_file}")