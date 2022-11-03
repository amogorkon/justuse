import use

FastAPI = use(
    "fastapi",
    version="0.85.2",
    modes=use.auto_install,
    hash_algo=use.Hash.sha256,
    hashes={
        "M䎹仏攡䪥䠌珥踄歕嬇䔟ȏ蛍㓤㖶㖣系㲯",  # None-None
        "P褸貦莋㟝縉擣樹鮗綞躋馐㛍㥂髟榛㐿牒",  # None-None
    },
)
print("ASDF")
