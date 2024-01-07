#!/usr/bin/env python
# coding: utf-8

# In[2]:


class Bank:
    def __init__(self, balance, pin):
        self.pin = pin
        self.balance = balance

    def withdraw(self, amt):
        atm_pin = int(input('Enter your ATM pin: '))
        if atm_pin == self.pin:
            if amt < self.balance:
                self.balance -= amt
                print(f'Withdrawn Rs. {amt}, current balance: {self.balance}')
            else:
                print('Invalid amount')
        else:
            print('Incorrect pin')

    def deposit(self, amt):
        atm_pin = int(input('Enter your ATM pin: '))
        if atm_pin == self.pin:
            self.balance += amt
            print(f'Deposited Rs. {amt}, current balance: {self.balance}')
        else:
            print('Incorrect pin')

    def check_balance(self):
        atm_pin = int(input('Enter your ATM pin: '))
        if atm_pin == self.pin:
            print(f'Current balance: {self.balance}')
        else:
            print('Incorrect pin')

bank_account = Bank(balance=1000, pin=1004)
bank_account.withdraw(500)
bank_account.deposit(200)
bank_account.check_balance()


# In[ ]:




