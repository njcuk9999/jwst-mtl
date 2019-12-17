## Requesting a Compute Canada account

To request a Compute Canada account, instructions are here:

[Asscessing-resources](https://www.computecanada.ca/research-portal/accessing-resources/)


Essentially, you first request for an account by clicking the “Register” link in the bottom right of this page:

[Compute Canada Login](https://ccdb.computecanada.ca/security/login)

Click “agree” on the first 2 forms and “yes” to the consent and finally “agree” to the terms of use. Then submit the form electronically.

In the next form, click no if this is the first time you apply for a CC account. Then fill your information.
```
Institution - Calcul Québec: Université de Montréal
Department - Departement De Physique
```
Then hit the Submit button.

You will need to confirm your user name for that account.

Next you will need (may have to be after they send you a confirmation email, I don't remember) to tell who your godfather is for the account. It is David Lafreniere: ```uns-623```

CC will wait for David to confirm you as a valid user. Ping him if there is no email from CC for a long time. Then, CC will send an email stating that you are a registered user.

## Using Compute Canada

The servers available to you are listed here:

[Available-Resources](https://www.computecanada.ca/research-portal/accessing-resources/available-resources/)

beluga is one available (in Quebec I suppose)
```bash
ssh lalbert@beluga.computecanada.ca 
```
(Use you user name, not the CC account identity such as emp-999)

Then you will need to read documentation in order to learn how to use python or other software on the server:

[Compute Canada Documentation](https://docs.computecanada.ca/wiki/Compute_Canada_Documentation)

For python, one needs to load modules:

[Python Modules](https://docs.computecanada.ca/wiki/Utiliser_des_modules/en)

A python-specific guide is here:

[Python Guide](https://docs.computecanada.ca/wiki/Python)

For example:
```bash
[lalbert@beluga4 ~]$ module load python/3.8 numpy
```
