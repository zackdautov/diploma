a = " Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
if "ipsum" in a:
    a = a.replace("ipsum", "", 1) + " ipsum"
print(a)