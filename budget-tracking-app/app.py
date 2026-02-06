class Person:
    def __init__(self, name, location, monthly_salary, total_expenditure, intent):
        self.name = name
        self.location = location
        self.monthly_salary = monthly_salary
        self.total_expenditure = total_expenditure
        self.intent = intent

    @property
    def savings(self):
        return self.monthly_salary - self.total_expenditure

    @property
    def spending_percentage(self):
        return (self.total_expenditure / self.monthly_salary) * 100

    @property
    def savings_percentage(self):
        return 100 - self.spending_percentage

    def assess(self):
        pct = self.spending_percentage
        savings = self.savings

        print(f"\n{'='*50}")
        print(f"  Budget Report for {self.name}")
        print(f"{'='*50}")
        print(f"  Location        : {self.location}")
        print(f"  Budgeting Goal  : {self.intent}")
        print(f"  Monthly Salary  : ${self.monthly_salary:,.2f}")
        print(f"  Total Spending  : ${self.total_expenditure:,.2f}")
        print(f"  Savings         : ${savings:,.2f}")
        print(f"  Spending        : {pct:.1f}% of income")
        print(f"  Saving          : {self.savings_percentage:.1f}% of income")
        print(f"{'='*50}")

        print("\n  Honest Assessment:")

        if savings < 0:
            print(f"  You are overspending by ${abs(savings):,.2f}.")
            print("  This is unsustainable. You need to cut expenses")
            print("  immediately or find additional income.")
        elif pct > 90:
            print("  You're spending over 90% of your income.")
            print("  One unexpected bill could put you in debt.")
            print("  Trim non-essentials aggressively.")
        elif pct > 75:
            print("  You're spending a lot - over 75% of your income.")
            print("  Your savings buffer is thin. Look for areas to")
            print("  reduce spending, especially discretionary costs.")
        elif pct > 50:
            print("  You're in a reasonable range but there's room to")
            print("  improve. The 50/30/20 rule suggests 50% on needs,")
            print("  30% on wants, 20% on savings. Review your wants.")
        elif pct > 30:
            print("  Solid financial discipline. You're saving a good")
            print("  chunk of your income. Keep it up and consider")
            print("  investing the surplus.")
        else:
            print("  Excellent. You're living well below your means.")
            print("  Make sure your savings are working for you -")
            print("  consider investments or retirement contributions.")

        print(f"{'='*50}\n")


def get_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("  Please enter a positive number.")
                continue
            return value
        except ValueError:
            print("  Invalid input. Enter a number.")


def main():
    print("\n  Budget Analyzer")
    print("  ---------------\n")

    name = input("  Enter your name: ").strip()
    location = input("  Enter your location: ").strip()
    intent = input("  What is your budgeting goal? (e.g., save for house, reduce debt): ").strip()
    salary = get_float("  Enter your monthly salary: $")
    if salary == 0:
        print("  Salary cannot be zero.")
        return
    expenditure = get_float("  Enter your total monthly expenditure: $")

    person = Person(name, location, salary, expenditure, intent)
    person.assess()


if __name__ == "__main__":
    main()
