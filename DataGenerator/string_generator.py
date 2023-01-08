import random as rnd
import string

def make_code_num(length):
    return "".join(rnd.choice("0123456789") for _ in range(length))

def make_num(max_length):
    num = ""
    for _ in range(rnd.randint(1,max_length)):
      num += str(rnd.randint(0,9))
    return num

def make_engl(max_length):
    eng = ""
    eng = eng.join(rnd.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(rnd.randint(1,max_length)))
    return eng

def make_viet(max_w, max_l, dic):  
    sample = rnd.sample(dic, rnd.randint(1,max_w))
    out = ""
    for s in sample:
      if len(out) + len(s) < max_l:
        out += s + " "
    return out[:-1]
    

def create_strings_from_dict(count, dic, text_scale):
    strings = []
    for _ in range(0, count):
        current_string  = make_viet(3, 35, dic)
        
        mode = rnd.randint(1,100)
        if mode < text_scale*100:
            #a
            mode2 = rnd.randint(1,7)
            #aa
            if mode2 == 1:
              output = current_string
            #AA 
            if mode2 == 2:
              output = current_string.upper()
            #Aa
            if mode2 == 3:
              output = current_string.title()
            #Aa 00g
            if mode2 == 4:
              output = current_string.title() + " " + make_num(3) + rnd.choice(["g", "kg", "G", "KG", "L", "M", "CM", "m", "%"])
            #Aa(a)
            if mode2 == 5: 
              spl = current_string.title().split(" ") 
              s1 = rnd.randint(0, len(spl)-1)
              s2 = rnd.randint(0, len(spl)-1)
              output = ""
              for (i, s) in enumerate(spl):
                if i == min(s1, s2):
                  s = "(" + s
                if i == max(s1, s2):
                  s = s + ")"
                output += s + " "
              output = output[:-1]
            #EE aa
            if mode2 == 6:
              output = make_engl(7) + " " + make_viet(3, 35, dic)
            #aa: aa
            if mode2 == 7:
              output = make_viet(3, 20, dic) + rnd.choice([": ","/"]) + make_viet(3, 20, dic)
        else:
            mode2 = rnd.randint(1,6)
            if mode2 == 1:
              # 1
              output = str(rnd.randint(1,30))
            if mode2 == 2:
              # 000,000đ
              output = '{:,}đ'.format(rnd.randint(0,10000)*1000)
            if mode2 == 3:
              # 000,123
              output = '{:,}'.format(rnd.randint(0,1000000))
            if mode2 == 4:
              # -000,123
              output = '-{:,}'.format(rnd.randint(0,1000000))
            if mode2 == 5:
              # 000.123
              output = '{:,}'.format(rnd.randint(0,1000) + rnd.randint(0,1000)/1000)
            if mode2 == 6:
              # 1x
              output = str(rnd.randint(1,30)) + rnd.choice(["X", "x", "g", "kg", "G", "KG", "L", "M", "CM", "m", "%"])

        if len(output) >45:
          print("pip")
          out = output.split(" ")
          output = ""
          for o in out:
            if len(output) + len(o) > 45:
              break
            output = output + o + " "
          output = output[:-1]
        strings.append(output)

    return strings
