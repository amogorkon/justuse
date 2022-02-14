from string import Template
import webbrowser

import base64

hashes = hashes={'G㻻䯹稷霯鍞躆䭽尡悴㶨腟㣴䪕夯懤幭㕒', 'K剉蒼繘䀇駈樿㷨霓谘挙豵㼮䗱鏳豣旝谎', '芵暻跻化浀䄥驦枟犲蟉㢏梤咪鐚ǃ劌矊', 'V鶚棓㖋㒙瞉纼䌉兮㵠認槠瀒ȱ舿䫾讲绂', '姎刼响忷䩮鑅䩵襐竰松㜰鮝堏皐峺鞝媲', 'H䬮㡕惼弧汶纫岽蚊諵趋䎒臒騿棦嘰溽旞', 'X漊谑澃潜沤荡浉歺诌硾㤂騻䋓狠䀚鴌菍', 'N軦崳䜃椨玀閶慽䊼蛄㚴ʃ嘩䉿㼾旀䋑伒', 'S龐脪郄鲊巗䩆货䌓䭏㜶ʦ鍰殙þ槯鶏㦏', 'N旅承脃厼躊桩䎿嗲㮤企燃䯼遜䏔笛夛砥', 'H䓨鮙䰜䪕识郉欴房媧誣邜爿埘炇撲鹉䈆', 'M处㶑帯㟝閲咤几帰䚆盌挩麻轵脱讚蚶舉', 'N㨀驁痻觖㱹㤺偳據驻藵䳐芳䅕簧窗㤔诱', 'O甚䰚豁䧑求晜讨錶䧡䥾翢低胤侗䔩嬒杵', 'P似鐒酂竁擬処殒垰薛拞㡅㓲㺓㼖丈薫帙', 'I褹㹵腪騑㼢ɭ㓤䃗㥜㾠剓伭㸇鍜䴠㲋赍', 'Q蟖㺘怇郸帺㹢麍鯾瑗笧汚涵䬳羍尥蟶备', 'g峯歰㷲灶樟秙嵎潪鑘㳚鸔䜸䛤悃耎銾䈈', 'T蟸玦蹰橨㚽嚉㱏郁㽋㻻癍霶剨抨挍㹻踢'}

template = Template("""<html><body>
    <h1>$package_name</h1>
    $hashes
    <p>
    Please specify the hash for auto-installation.
    </p>
    <p>
    </p>
    </body></html>""")

package_name = "foo"
html = template.substitute(**globals())