import use

mod = use(use.Path(".test4.py"))

mod @ use.woody_logger

mod.test1(2)
mod._test2(2)

t = mod.Test(2)
t.test3()

use.show_aspects()
