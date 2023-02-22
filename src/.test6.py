import use

load = use(
    use.URL("https://raw.githubusercontent.com/amogorkon/stay/master/src/stay/stay.py"), modes=use.recklessness
, import_as="stayy").Decoder()

load("a: b")