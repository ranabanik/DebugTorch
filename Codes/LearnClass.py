"rana.banik@vanderbilt.edu"
if __name__ != '__main__':
    class User:
        pass #do nothing

    user1 = User()  # user1 is called: object
    print(user1)

    user1.first_name = "Rana"       # first_name is called: field
    user1.last_name = "Banik"

    print(user1.first_name)  # Rana

import datetime
if __name__ == '__main__':
    class User:
        def __init__(self, full_name, birthday):     # initialization or in some other language: constructor
            #   docstring
            """
            first argument of init is self
            self.field = value

            The field is attached to the object itself"""
            self.name = full_name
            self.birthday = birthday    #yyyymmdd

            # lets add another feature to the class
            name_pieces = full_name.split(" ")
            self.first_name = name_pieces[0]
            self.last_name = name_pieces[-1]

        def age_method(self):     # another method to the class
            """
            is actually functions of class called methods.
            Like init here the first argument is self
            """
            today = datetime.date(2020, 4, 20)
            yyyy = int(self.birthday[0:4])
            mm = int(self.birthday[4:6])
            dd = int(self.birthday[6:])
            dob = datetime.date(yyyy, mm, dd)
            age_in_days = (today - dob).days
            age_in_years = age_in_days / 365
            return int(age_in_years)

    rana_object = User("Rana Banik", "19920309") # created an instance
    print(rana_object.age_method())
    #   self wasn't called when method called. Only used when writing the class structure
