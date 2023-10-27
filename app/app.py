from appmodels import load_best_model, prep_data_from_user
from flask import Flask, flash, render_template, request

print("Starting app...")
print(
    """

    -----------------------------------------------------------
      

                        $"   *.      
                    d$$$$$$$P"                  $    J
                        ^$.                     4r  "
                        d"b                    .db
                    P   $                  e" $
            ..ec.. ."     *.              zP   $.zec..
        .^        3*b.     *.           .P" .@"4F      "4
        ."         d"  ^b.    *c        .$"  d"   $         %
    /          P      $.    "c      d"   @     3r         3
    4        .eE........$r===e$$$$eeP    J       *..        b
    $       $$$$$       $   4$$$$$$$     F       d$$$.      4
    $       $$$$$       $   4$$$$$$$     L       *$$$"      4
    4         "      ""3P ===$$$$$$"     3                  P
    *                 $       ""m        b                J
        ".             .P                    %.             @
        %.         z*"                      ^%.        .r"
            "*==*""                             ^"*==*""   

      
    WELCOME TO THE CYCLE TRAFFIC PREDICTING WEBSITE!
        
    PLEASE BE PATIENT IF THIS IS THE FIRST RUN AS THE MODEL NEEDS TO BE BUILT FROM SCRATCH
        
    AFTER INTIAL BUILD, THE MODEL CAN BE USED WITH NO WAIT TIME!
      
    -----------------------------------------------------------
    """
)

app = Flask(__name__)
app.secret_key = "A_in_INF161?_:P"

predictor = load_best_model()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(f" INPUT : {input_dict}")
        prepped_data = prep_data_from_user(input_dict)
        print(f" INPUT: PREPPED DATA = {prepped_data}")

        if isinstance(prepped_data, str):
            flash(
                f"ERROR: Ensure all inputs types are numbers, and the date is in the proper format"
            )
            return render_template("home.html")

        trafficamount = predictor.predict(prepped_data)

        # bruk floor her TODO
        trafficamount = int(trafficamount[0])

        cyclist = f"""     
                            _______________   
                          / Hei Nello :^)     
                         <  I am one of {trafficamount}!   >
                    __o   \_________________/
                 _ |/<_
                (_)| (_)

                """

        cyclist_row = " " * 20  # Specifies the distance between each cyclist
        cycle_art = (cyclist + cyclist_row) * trafficamount

        return render_template(
            "home.html", traffic_data=trafficamount, cycle_art=cycle_art
        )

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, port="8080")
