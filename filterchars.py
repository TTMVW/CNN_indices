import sys
import re



class Descriptor(object):
    def __init__(self, short_code: str, raw: str):
        self.code = short_code
        self.raw = raw
        self.learning_outcomes = None

        self.aim = None
        self.level =  None
        self.content = None

        self.full_name = None

        self.prerequisites = None

        self.co_requisites = None

    def set_learning_outcomes(self, fn):
        self.learning_outcomes = fn(self.raw)

    def set_content(self, fn):
        self.content = fn(self.raw)

    def set_aim(self, fn):
        self.aim = fn(self.raw)

    def set_full_name(self, fn):
        self.full_name = fn(self.raw)

    def set_pre_requisites(self, fn):
        self.prerequisites = fn(self.raw)

    def set_co_requisites(self, fn):
        self.co_requisites = fn(self.raw)


def get_txt_between(raw: str, re1: re, re2: re):
    whole_list = re.split(re1, raw)
    with_All = ""
    for index in range(1, len(whole_list)):
        with_All += whole_list[index]
    the_text = (re.split(re2, with_All))[0]
    return the_text


def MIT():
    # pdf2txt -n ./'MIT Course Descriptors.pdf' > origin.txt
    # Read txt into courses
    # pdf2txt -n ./'MIT Course Descriptors.pdf' | /usr/bin/python3 ../filterchars.py  > first_courses.txt MIT
    course_content = {}
    instr = sys.stdin.read()
    courses = re.split(r" Course title", instr)
    for course in courses:
        clean = re.sub("", "", course)
        long_name_re = re.search(r"^.*Course code", clean)
        long_name = (long_name_re.group())[:-12]
        if " Course title " in long_name:
            long_name = long_name[14:]
        long_name = long_name.strip()
        code = get_txt_between(clean, r"Course code", r"Directed learning").strip()
        if len(code) > 7 :
            code = get_txt_between(clean, r"Course code", r"Work integrated").strip()
        descriptor = Descriptor(code, clean)
        descriptor.full_name = long_name
        course_content[code] = descriptor
        # 
        re1 = r"Learning outcomes "
        re2 = r"Contents?"
        descriptor.learning_outcomes = get_txt_between(clean, re1, re2).strip()
        #
        re1 = r"Aim "
        re2 = r"Learning outcomes"
        descriptor.aim = get_txt_between(clean,re1,re2).strip()
        #
        re1 = r"Content "
        re2 = r"Assessment Number"
        descriptor.content = get_txt_between(clean,re1,re2).strip()
        #
        re1 = r"Pre-requisites "
        re2 = r"Co-requisites "
        descriptor.prerequisites = get_txt_between(clean,re1,re2).strip()
        #
        re1 = r"Co-requisites "
        re2 = r"Attendance requirements "
        descriptor.co_requisites = get_txt_between(clean,re1,re2).strip()
        #
        re1 = r"Level"
        re2 = r"Total Learning Hours"
        descriptor.level = get_txt_between(clean,re1,re2).strip()
        
        

    return course_content
    pass


def NMIT():
    #  pdf2txt -n ./'NMIT BIT Course Descriptors 2021 08221.pdf' > original.txt
    #  pdf2txt -n ./'NMIT BIT Course Descriptors 2021 08221.pdf' | /usr/bin/python3 ../filterchars.py 
    def get_long_name(clean):
        re1 = r"" + short_name
        re2 = r"Version Effective"

        return get_txt_between(clean, re1, re2).strip()

    def get_descriptor(course_content, short_name, clean):
        long_name = get_long_name(clean)

        # print(short_name, long_name, clean)
        descriptor = Descriptor(short_name, clean)

        re1 = r"LEARNING OUTCOMES"
        re2 = r"ASSESSMENTS"
        descriptor.learning_outcomes = get_txt_between(clean, re1, re2).strip()

        re1 = r"Indicative c?C?ontent"
        re2 = r"LEARNING OUTCOMES"
        descriptor.content = get_txt_between(clean, re1, re2).strip()
        if len(descriptor.content) == 0:
            re2 = "REQUIREMENTS FOR SUCCESSFUL COURSE"
            descriptor.content= get_txt_between(clean, re1, re2).strip()
         

        re1 = r"Course a?A?im"
        if len(descriptor.content) == 0 :
            re2 = r"ASSESSMENTS|LEARNING OUTCOMES"
        else:
            re2 = r"Indicative c?Content"
           
        descriptor.aim = get_txt_between(clean, re1, re2).strip()

        re1 = r"Pre-requisites"
        re2 = r"Co-requisites"
        descriptor.prerequisites = get_txt_between(clean, re1, re2).strip()

        re1 = r"Co-requisites"
        re2 = r"Alignment"
        descriptor.co_requisites = get_txt_between(clean, re1, re2).strip()

        descriptor.full_name = short_name + ' ' + long_name
        course_content[short_name] = descriptor
        
        re1 = r"NMIT c?C?redits [0-9][0-9] Level "
        re2 = r" EFTS"
        descriptor.level = get_txt_between(clean, re1, re2).strip()
        
            

    course_content = {}  # A dictionary of "Descriptors" by course

    # Read txt into courses
    instr = sys.stdin.read()
    courses = re.split(r"([' ']+[A-Z]{3}[0-9]{3})", instr)
    first_course = True
    count = 1
    short_name = ""
    long_name = ""
    for course in courses:
        clean = re.sub(r"", "", course).strip()

        if first_course:
            re_result = re.search(r"^[' ']*[A-Z]{3}[0-9]{3}", clean)
            short_name = re_result.group().strip()
            get_descriptor(course_content, short_name, clean)

            first_course = False
        else:
            if count % 2 == 0:
                short_name = clean
            else:
                long_clean = short_name + ' ' + clean
                get_descriptor(course_content, short_name, long_clean)

        count += 1

    return course_content


def NorthTec():
    # pdf2txt -n ./"NorthTec Structure.Descriptors 2019.pdf" > new_origin.txt
    # THEN Edit new origin to leave the descriptors
    # mv new_origin.txt new_origin_do_not_edit.txt
    # /usr/bin/python3 ../filterchars.py < new_origin_do_not_edit.txt > clean1.txt
    # /usr/bin/python3 ../filterchars.py < new_origin_do_not_edit.txt > code_name.txt
    # /usr/bin/python3 ../filterchars.py < new_origin_do_not_edit.txt > code_name.txt

    def get_long_name(prev_course):

        x = re.search("LEARNING AND TEACHING RESOURCES([' '][A-Z/-].+)([A-Z]|[0-9])$", prev_course)
        if x is None:
            x = re.search("LEARNING AND TEACHING RESOURCES[' ']*([' '][A-Z/-].+)([A-Z]|[0-9])$", prev_course)
            y = x.group()
        else:
            y = x.group()

        z = re.search("([' ']?[A-Z]+)*([A-Z]|[' '][0-9])$", y)
        result = z.group().strip()
        return result

    def get_short_name(raw):
        result = (get_txt_between(raw, r"Course Code:", r"(Effective from)|(Inactive)")).strip()
        result = result.replace("Elective", "")
        result = result.replace("Compulsory", "")
        return result.strip()

    def assign_descriptor_from(course_content, long_name, clean):
        raw = long_name + " Course Code: " + clean

        short_name = get_short_name(raw)
        # print(short_name)
        # print(raw)
        a_descriptor = Descriptor(short_name, raw)
        a_descriptor.full_name = long_name

        # Get Aim 
        re1 = r"COURSE AIM"
        re2 = r"LEARNING OUTCOMES"
        a_descriptor.aim = get_txt_between(raw, re1, re2)

        # Get Learning outcomes

        re1 = r"LEARNING OUTCOMES"
        re2 = r"TOPICS / INDICATIVE CONTENT"
        a_descriptor.learning_outcomes = get_txt_between(raw, re1, re2)

        # Get Content
        re1 = r"INDICATIVE CONTENT"
        re2 = r"ASSESSMENT"
        a_descriptor.content = get_txt_between(raw, re1, re2)

        # Get Prerequisites
        re1 = r"Pre-requisites:"
        re2 = r"Co-requisites:"
        a_descriptor.prerequisites = get_txt_between(raw, re1, re2)

        # Get Corequisites
        re1 = r"Co-requisites:"
        re2 = r"COURSE AIM"
        a_descriptor.co_requisites = get_txt_between(raw, re1, re2)

        re1 = r"Co-requisites:"
        re2 = r"COURSE AIM"
        a_descriptor.level= get_txt_between(raw, re1, re2)
        
        # Put the new descriptor in to course_content
        course_content[short_name] = a_descriptor

        return raw
        #pass

    course_content = {}  # A dictionary of "Descriptors" by course

    # Read txt into courses
    instr = sys.stdin.read()

    # Remove FF
    clean = re.sub(r"", "", instr).strip()
    # Remove Page N of N[N]?
    cleaner = re.sub(r"Page [0-9][0-9]? of [0-9][0-9]?", "", clean).strip()

    course_code_split = cleaner.split('Course code:')

    first_item = True
    first_course = True
    prev = None
    long_name = None
    for a in course_code_split:
        clean = a.strip()

        if first_item:
            long_name = clean

            first_item = False

        elif first_course:
            prev = assign_descriptor_from(course_content, long_name, clean)
            first_course = False

        else:
            long_name = get_long_name(prev)
            prev = assign_descriptor_from(course_content, long_name, clean)

            # prev = clean # cleaned
            # first_course = False;

    # for a in course_code_split:
    #     clean = a.strip()
    #     if not first: 
    #         # The course name is the last part of the SMS CODE split, preceeded by "creativecommons" etc
    #         list = (re.split(r"http://creativecommons.org/licenses/by/3.0/nz/ [0-9][0-9]?",prev))
    #         name = list[len(list)-1].strip()
    #         cleaner = re.sub(r"Programme Document:   Approved: Academic Board 22 July 2009 OT4765 Bachelor of Information Technology Amended Academic Board 08 October 2019  http://creativecommons.org/licenses/by/3.0/nz/ [0-9][0-9]?","",clean)

    #         # Reconstruct "raw"  
    #         raw = name + " SMS Code " + cleaner

    #         # Get short_name
    #         re1 = r"SMS Code"
    #         re2 = r"Directed Learning"
    #         short_name = (get_txt_between(raw,re1,re2)).strip()

    #         # Get Learning Outcomes 
    #         re1 = r"Learning Outcomes"
    #         re2 = r"Indicative Content"
    #         learning_outcomes = (get_txt_between(raw,re1,re2)).strip() 

    #         # Get Content 
    #         re1 = r"Indicative Content"
    #         re2 = r"Assessment"
    #         indicative_content = (get_txt_between(raw,re1,re2)).strip() 

    #         # Get Aim
    #         re1 = r"Aims"
    #         re2 = r"Learning Outcomes"
    #         aim = (get_txt_between(raw,re1,re2)).strip() 

    #         #Get Prerequisites
    #         re1 = r"Prerequisites"
    #         re2 = r"Total Learning Hours"
    #         prerequisites= (get_txt_between(raw,re1,re2)).strip() 

    #         # Create a descriptor with this as "raw"
    #         descriptor = Descriptor(short_name,raw)
    #         descriptor.full_name = name
    #         descriptor.learning_outcomes = learning_outcomes
    #         descriptor.content = indicative_content
    #         descriptor.aim = aim
    #         descriptor.prerequisites = None if prerequisites == "n/a" else prerequisites 
    #         descriptor.co_requisites = None
    #         course_content[short_name] = descriptor 

    #         # Clean up prev raw
    #         if prev_name != None:
    #             course_content[prev_name].raw = course_content[prev_name].raw.replace(name,"")

    #             prev_name = short_name
    #         else:
    #             prev_name = short_name  

    #         prev = clean

    #     else :
    #        prev = clean
    #        first = False

    #     # Clean up raw
    #     # for name in course_content:

    return course_content

    pass


def Otago_BIT():
    # Run with BASH
    # cd Otago
    # Get the orginal
    # cd Otago
    # pdf2txt -n ./'Otago OT4765 OT4978 OT4979 Bach-GdCert-GdDip Descriptors 2021 Final.pdf'  > otago_org.txt
    # Programme Document:   Approved: Academic Board 22 July 2009 OT4765 Bachelor of Information Technology Amended Academic Board 08 October 2019  http://creativecommons.org/licenses/by/3.0/nz/(( [0-9] [0-9][.][0-9][.])|( )).*SMS Code
    # pdf2txt -n ./'Otago OT4765 OT4978 OT4979 Bach-GdCert-GdDip Descriptors 2021 Final.pdf' | ../filterchars.py
    # pdf2txt -n ./'Otago OT4765 OT4978 OT4979 Bach-GdCert-GdDip Descriptors 2021 Final.pdf' | /usr/bin/python3 ../filterchars.py

    course_content = {}  # A dictionary of "Descriptors" by course
    # Read txt into courses
    instr = sys.stdin.read()
    sms_split = instr.split('SMS Code')

    first = True
    prev = None
    prev_name = None
    for a in sms_split:
        clean = re.sub(r"", "", a).strip()
        if not first:
            # The course name is the last part of the SMS CODE split, preceeded by "creativecommons" etc
            list = (re.split(r"http://creativecommons.org/licenses/by/3.0/nz/ [0-9][0-9]?", prev))
            name = list[len(list) - 1].strip()
            cleaner = re.sub(
                r"Programme Document:   Approved: Academic Board 22 July 2009 OT4765 Bachelor of Information Technology Amended Academic Board 08 October 2019  http://creativecommons.org/licenses/by/3.0/nz/ [0-9][0-9]?",
                "", clean)

            # Reconstruct "raw"  
            raw = name + " SMS Code " + cleaner

            # Get short_name
            re1 = r"SMS Code"
            re2 = r"Directed Learning"
            short_name = (get_txt_between(raw, re1, re2)).strip()

            # Get Learning Outcomes 
            re1 = r"Learning Outcomes"
            re2 = r"Indicative Content"
            learning_outcomes = (get_txt_between(raw, re1, re2)).strip()

            # Get Content 
            re1 = r"Indicative Content"
            re2 = r"Assessment"
            indicative_content = (get_txt_between(raw, re1, re2)).strip()

            # Get Aim
            re1 = r"Aims"
            re2 = r"Learning Outcomes"
            aim = (get_txt_between(raw, re1, re2)).strip()

            # Get Prerequisites
            re1 = r"Prerequisites"
            re2 = r"Total Learning Hours"
            prerequisites = (get_txt_between(raw, re1, re2)).strip()

            # Create a descriptor with this as "raw"
            descriptor = Descriptor(short_name, raw)
            descriptor.full_name = name
            descriptor.learning_outcomes = learning_outcomes
            descriptor.content = indicative_content
            descriptor.aim = aim
            descriptor.prerequisites = None if prerequisites == "n/a" else prerequisites
            descriptor.co_requisites = None
            course_content[short_name] = descriptor

            # Clean up prev raw
            if prev_name != None:
                course_content[prev_name].raw = course_content[prev_name].raw.replace(name, "")

                prev_name = short_name
            else:
                prev_name = short_name

            prev = clean

        else:
            prev = clean
            first = False

        # Clean up raw
        # for name in course_content:

    return course_content
    pass


def PR5006_HV4701_BIT():
    #  Run with BASH
    #  pdf2txt -n ./PR5006-HV4701_BIT_Programme_descriptors.pdf | /usr/bin/python3 ./filterchars.py
    def get_learning_outcomes(raw: str):
        # scan raw to accumulate learning outcomes
        # return LOs
        re1 = r"(?i)Learning Outcomes[:]?"
        re2 = r"(?i)Indicative Content[:]?"
        return (get_txt_between(raw, re1, re2)).strip()

    def get_content(raw: str):
        # scan raw to accumulate content
        # return content

        re1 = r"(?i)[Cc]ontent"
        re2 = r"(?i)Assessment Method"
        return get_txt_between(raw, re1, re2)

    def get_aim(raw: str):
        re1 = r"(?i)Aim[s]?"
        re2 = r"(?i)Learning Outcomes"
        return get_txt_between(raw, re1, re2)

    def get_full_name(raw: str):
        re1 = r"(?i)Title"
        re2 = r"(?i)Level"
        result = get_txt_between(raw, re1, re2)
        return result.strip()

    def get_pre_requisites(raw: str):
        re1 = r"(?i)Pre-requisites"
        re2 = r"(?i)Learning Hours"
        result = get_txt_between(raw, re1, re2)
        return result.strip()

    """ No Co_requisites - beware the following
        def get_co_requisites(raw:str):
            Pass 
            return ""
            """

    course_content = {}  # A dictionary of "Descriptors" by course
    # Read txt into pages
    instr = sys.stdin.read()
    page_list = instr.split('')

    current_module = ""
    raw_list = []
    # start match
    re_start = r".*Code Title [A-Z]{2}[0-9]{4}"
    for page in page_list:
        proposed_raw = (
            re.sub(r"PR5006 Bachelor of Information Technology  page [0-9][0-9]?[0-9]?", "", page.strip())).strip()
        if proposed_raw != "":
            start_of_module = re.match(r"^[A-Z]{2}[0-9]{4}", proposed_raw)
            if start_of_module is not None:
                current_module = start_of_module.group()
                course_content[current_module] = Descriptor(current_module, proposed_raw)
            else:
                course_content[current_module].raw += page
    # Process raw
    for descriptor_code in course_content:
        course_content[descriptor_code].set_learning_outcomes(get_learning_outcomes)
        course_content[descriptor_code].set_content(get_content)
        course_content[descriptor_code].set_aim(get_aim)
        course_content[descriptor_code].set_full_name(get_full_name)
        course_content[descriptor_code].set_pre_requisites(get_pre_requisites)

    return course_content


def Ucol_Bachelor_of_Information_and_Communications_Technology_L7_Courses() -> dict:
    # Ucol get descriptors
    # Run with Bash command:
    # pdf2txt -n ./'UCOL Bachelor of Information and Communications Technology L7 Courses.pdf' | /usr/bin/python3 ./filterchars.py > UCOL.txt

    def get_learning_outcomes(raw: str):
        # scan raw to accumulate learning outcomes
        # return LOs
        re1 = r"(?i)Learning Outcomes[:]?"
        re2 = r"(?i)Content[:]?"
        return (get_txt_between(raw, re1, re2)).strip()

    def get_content(raw: str):
        # #scan raw to accumulate content
        # #return content

        re1 = r"(?i)Content[:]?"
        re2 = r"(?i)Learning and Teaching[:]?"
        return get_txt_between(raw, re1, re2)

    def get_aim(raw: str):
        re1 = r"(?i)Course Aim[:]?"
        re2 = r"(?i)Learning Outcomes[:]?"
        return (get_txt_between(raw, re1, re2)).strip()

    def get_full_name(raw: str):
        re1 = r"[A-Z][0-9]{3}"
        re2 = r"(?i)Course Level[:]?"
        result = get_txt_between(raw, re1, re2)
        return result.strip()

    def get_pre_requisites(raw: str):
        result = ""
        re1 = r"(?i)Pre-requisite or Co-requisite"
        re2 = r"(?i)Course Aim"
        result = get_txt_between(raw, re1, re2)
        if result == "":
            re1 = r"(?i)Pre-requisite"
            re2 = r"(?i)Co-requisite"
            result = get_txt_between(raw, re1, re2)
        return result.strip()

    def get_co_requisites(raw: str):
        re1 = r"(?i)Co-requisite"
        re2 = r"(?i)Course Aim"
        result = get_txt_between(raw, re1, re2)
        return result.strip()

    course_content = {}  # A dictionary of "Descriptors" by course

    # Read txt into pages
    instr = sys.stdin.read()
    page_list = instr.split('')

    current_module = ""

    # Get descriptors - filter out page and start of modules, process for each descriptor 
    for page in page_list:

        # Bachelor of Information and Communications Technology Level 7 Version 21.2 Approved by: NZQA  Page 1 of 76 Master Copy: I/CAS/curriculum documents and programme file   
        repagestr = r"Bachelor.*Page [0-9]+ of [0-9]+ Master.*programme file"
        restart = r"[A-Z][0-9]{3}.*Course Level"
        proposed_output = (re.sub(repagestr, "", page)).strip()
        # test_filter_pages += [filtered_str]

        if proposed_output != "":
            # If a new course code
            the_module_match = re.match(r"[A-Z][0-9]{3}", proposed_output)
            if the_module_match is not None:
                current_module = the_module_match.group()
                # Create a descriptor with this proposed_output as "raw"
                descriptor = Descriptor(current_module, proposed_output)
                course_content[current_module] = descriptor
            else:
                # append the proposed output to raw
                course_content[current_module].raw += proposed_output

    # Process "RAW" data 
    for descriptor_code in course_content:
        course_content[descriptor_code].set_learning_outcomes(get_learning_outcomes)
        course_content[descriptor_code].set_content(get_content)
        course_content[descriptor_code].set_aim(get_aim)
        course_content[descriptor_code].set_full_name(get_full_name)
        course_content[descriptor_code].set_pre_requisites(get_pre_requisites)
        course_content[descriptor_code].set_co_requisites(get_co_requisites)

    return course_content


def Unitec_BSC_Prog_Descriptors():  # -> dict :
    # Unitec get course descriptors
    # Unitec process
    # Based on https://www.digitalocean.com/community/tutorials/how-to-perform-server-side-ocr-on-pdfs-and-images
    # BASH gs -o unitec_output/%05d.png -sDEVICE=png16m -r300 -dPDFFitPage=true 'Unitec BCS Prog Descriptors.pdf'
    # BASH for png in $(ls unitec_output); do tesseract -l eng unitec_output/$png unitec_output/$(echo $png | sed -e "s/\.png//g") pdf; done
    # BASH pdftk unitec_output/*.pdf cat output unitec_output/joined.pdf
    # Then Run with Bash command
    # pdf2txt ./unitec_output/joined.pdf 
    # Still have a mess with OCR'd text moving on to UCOL for now
    # 9/March/20223 With a new source
    # Run with Bash command
    # cd ./Unitec
    # pdf2txt -n ./'UNITEC Course Descriptor Sem2-2022 v2 .pdf'  | /usr/bin/python3 ../filterchars.py

    def get_learning_outcomes(raw: str):
        # scan raw to accumulate learning outcomes
        # return LOs
        re1 = r"(?i)Learning Outcomes[:]?"
        re2 = r"(?i)Learning and Teaching[:]?"

        learning_outcome = (get_txt_between(raw, re1, re2)).strip()
        return learning_outcome

    def get_content(raw: str):
        # No indicative content - outcome statement is "big"  

        # re1 = r"(?i)Content[:]?"
        # re2 = r"(?i)Learning and Teaching[:]?"

        return ""

    def get_aim(raw: str):
        # The outcome statement is an aim
        re1 = r"(?i)Outcome Statement[:]?"
        re2 = r"(?i)Learning Outcomes[:]?"
        result = (get_txt_between(raw, re1, re2)).strip()
        if re.search("Course requirements:", result):
            re3 = r"(?i)Course requirements[:]?"
            result = (get_txt_between(raw, re1, re3)).strip()
        return result

    def get_full_name(raw: str):
        # re1 = r"^.*[?"
        # re2 = r"(?i)Course Number[:]?"
        result = re.split(r"Course number:", raw)[0]

        new_str = result[0]

        state = 0
        therest = result[1:]
        for c in therest:
            if state == 0:
                if c != ' ':
                    state = 1
                    new_str += c
            elif state == 1:
                new_str += c

        return new_str.strip()

    def get_pre_requisites(raw: str):

        result = ""
        re1 = r"(?i)Requisites / Restrictions:"
        re2 = r"(?i)Delivery mode:"
        whole_field = get_txt_between(raw, re1, re2)
        result = ','.join(re.findall(r"([A-Z]{4}[0-9]{4})", whole_field))

        return result.strip()

    def get_co_requisites(raw: str):
        # not clear how these work - to be re-considered, right now they are all in the prerequisites area
        result = ""
        return result.strip()

    course_content = {}  # A dictionary of "Descriptors" by course

    # Read txt into pages
    instr = sys.stdin.read()
    # page_list = instr.split('')
    # Unitec has in consistent FF page delimiter from the pdf2txt
    # But it has a unique course start identifier
    # going straight to descriptors
    list_name_course = list(re.split(r"([A-Z]{4}[0-9]{4}):", instr))
    short_name = ""
    i = 0
    for c in list_name_course:
        if (i % 2) != 0:
            # print(i,list_name_course[i],list_name_course[i+1][:10])
            short_name = list_name_course[i]
            clean_raw = re.sub(r"", "", list_name_course[i + 1]).strip()
            course_content[short_name] = Descriptor(short_name, clean_raw)
        i += 1

    # Process "RAW" data 
    for descriptor_code in course_content:
        course_content[descriptor_code].set_learning_outcomes(get_learning_outcomes)
        course_content[descriptor_code].set_content(get_content)
        course_content[descriptor_code].set_aim(get_aim)
        course_content[descriptor_code].set_full_name(get_full_name)
        course_content[descriptor_code].set_pre_requisites(get_pre_requisites)
        course_content[descriptor_code].set_co_requisites(get_co_requisites)

    # course_content
    return course_content


def Wintec_BAppliedIT_Vol2() -> dict:
    # Wintec - get descriptors
    # Run with Bash command
    # pdf2txt -n ./'Wintec BAppliedIT(Vol2) (ModDesc).pdf' | /usr/bin/python3 ../filterchars.py > Wintec.txt

    def get_learning_outcomes(raw: str):
        # scan raw to accumulate learning outcomes
        # return LOs
        re1 = r"(?i)Learning Outcomes[:]?"
        re2 = r"(?i)Content[:]?"
        return get_txt_between(raw, re1, re2)

    def get_content(raw: str):
        # scan raw to accumulate content
        # return content

        re1 = r"(?i)Content[:]?"
        re2 = r"(?i)Teaching Learning Methods[:]?"
        return get_txt_between(raw, re1, re2)

    def get_aim(raw: str):
        re1 = r"(?i)Aim[:]?"
        re2 = r"(?i)Learning Outcomes[:]?"
        return get_txt_between(raw, re1, re2)

    def get_full_name(raw: str):
        re1 = r"(?i)Module Name[:]?"
        re2 = r"(?i)Module Code[:]?"
        re3 = r"[A-Z]{4}[0-9]{3} [–]?"
        re4 = r"(?i)Credit Value[:]?"
        result = get_txt_between(raw, re1, re2)
        if (result == "") or (result is None):
            result = get_txt_between(raw, re3, re4)
        return result.strip()

    def get_pre_requisites(raw: str):
        re1 = r"(?i)Pre-Requisites[:]?"
        re2 = r"(?i)Co-Requisites[:]?"
        return get_txt_between(raw, re1, re2)

    def get_co_requisites(raw: str):
        result = ""
        re1 = r"(?i)Co-requisites[:]?"

        re2 = r"(?i)Aim[:]?"
        mode = get_txt_between(raw, re1, re2)
        if mode == None:
            re2 = r"(?i)Mode of Delivery[:]?"
            result = get_txt_between(raw, re1, re2)
        else:
            result = mode
        return result
    
    def get_level(descriptor_code):
    
        the_text = descriptor_code[-3:]
        return the_text

    course_content = {}  # A dictionary of "Descriptors" by course

    # Read txt into pages
    instr = sys.stdin.read()
    page_list = instr.split('')

    current_module = ""

    # Get descriptors - filter out page and start of Wintec modules, process for each descriptor 
    for page in page_list:
        repagestr = r"[0-9][0-9] \| Page © Copyright 2015, Waikato Institute of Technology"
        restart = r" MODULE DESCRIPTOR FOR:  [A-Z]{4}[0-9]{3}"
        if page.startswith(' MODULE DESCRIPTOR FOR:  '):

            proposed_output = re.sub(r"^1\s*", "", (
                re.sub(
                    restart, "", (
                        re.sub(repagestr, "", page)
                    )
                )
            ).strip()
                                     )

            if proposed_output != "":
                # If a new course code
                the_module_match = re.match(r"[A-Z]{4}[0-9]{3}", proposed_output)
                if the_module_match is not None:
                    current_module = the_module_match.group()
                    # Create a descriptor with this proposed_output as "raw"
                    descriptor = Descriptor(current_module, proposed_output)
                    course_content[current_module] = descriptor
                else:
                    # append the proposed output to raw
                    course_content[current_module].raw += proposed_output

    # Process "RAW" data 
    for descriptor_code in course_content:
        course_content[descriptor_code].set_learning_outcomes(get_learning_outcomes)
        course_content[descriptor_code].set_content(get_content)
        course_content[descriptor_code].set_aim(get_aim)
        course_content[descriptor_code].set_full_name(get_full_name)
        course_content[descriptor_code].set_pre_requisites(get_pre_requisites)
        course_content[descriptor_code].set_co_requisites(get_co_requisites)
        course_content[descriptor_code].level = get_level(descriptor_code)

    return course_content

course_filters = {"NMIT":NMIT,"MIT": MIT,"NorthTec":NorthTec,
                    "Otago":Otago_BIT,"WandW":PR5006_HV4701_BIT,
                    "Ucol":Ucol_Bachelor_of_Information_and_Communications_Technology_L7_Courses,
                    "Unitec":Unitec_BSC_Prog_Descriptors, "WinTec":Wintec_BAppliedIT_Vol2
                    
} 
def show_all_of_program(pFilterKey):
    course_content = course_filters[pFilterKey]()
    for course in course_content:
        print(course_content[course].code)
        print(course_content[course].full_name)
        print("LOs\n",course_content[course].learning_outcomes)
        print("AIM\n",course_content[course].aim)
        print("Content\n",course_content[course].content)
        print("Prerequisites\n",course_content[course].prerequisites)
        print("Co-requisites\n",course_content[course].co_requisites)
        print("=====")
        
def show_columns(pFilterKey, pColumnNameList) :
    course_content = course_filters[pFilterKey]()
    first_line = True
    for course in course_content:
        aline = ""
        #if len(course_content[course].__dict__["content"]) == 0:
        first_column = True
        header = ""
        for columnName in pColumnNameList:  
               # print( columnName, course_content[course].__dict__[columnName],  len(course_content[course].__dict__[columnName]))
               if first_line:
                   header +=  int(not(first_column))*','+columnName
                   
               aline += int(not(first_column))*','+course_content[course].__dict__[columnName] #"[]"  str(len(course_content[course].__dict__[columnName]))\
               first_column = False
        if first_line:
            print(header)
        print(aline)
        first_line= False
        
        
    
if __name__ == "__main__":
    # 3,HVP WandW, OK, complete
    # 4	MIT,OK,	complete
    # 5,NMIT, OK,complete
    # 6,NorthTec, OK,complete
    # 8,Otago, OK,complete
    # 9,UCol, OK, complete
    # 10,UniTec, OK,complete
    # 11,WinTec,OK,complete
    
    # test code

    if len(sys.argv) > 0 and sys.argv[1] in list(course_filters.keys()):
        #show_all_of_program(sys.argv[1])
        show_columns(sys.argv[1],['code','level','learning_outcomes'])
    else:
        print("Require a specified program. One of ", list(course_filters.keys()))
            
        
    # NMIT
    # course_content = NMIT()
    # for a_course in course_content:
    #     print(course_content[a_course].code,
    #           course_content[a_course].full_name,
    #           # course_content[a_course].raw,
    #           course_content[a_course].learning_outcomes,
    #           course_content[a_course].content,
    #           course_content[a_course].prerequisites,
    #           course_content[a_course].co_requisites,
    #           )
    # NorthTec
    # course_content = NorthTec()
    # for course_code in course_content:
    #     print ( course_content[course_code].code,"::",
    #             course_content[course_code].full_name, "::AIM::",
    #             course_content[course_code].aim, "::LEARNING OUTCOMES::",
    #             course_content[course_code].learning_outcomes, "::CONTENT::",
    #             course_content[course_code].content, "::PREREQUISITES::",
    #             course_content[course_code].prerequisites, "::CO_REQUISITES::",
    #             course_content[course_code].co_requisites, "::",
    #            )
    # Otago
    # Spliting by SMS Code
    #    course_content = Otago_BIT()
    #    for course_code in course_content:
    #        print(course_code,">>>>>><<Full name>>",
    #             course_content[course_code].full_name,"<<Aim>>",
    #             course_content[course_code].aim,"<<Learning Outcome>>",
    #             course_content[course_code].learning_outcomes,"<<Content>>",
    #             course_content[course_code].content,"<<Prerequisites>>",
    #             course_content[course_code].prerequisites,"<<Co_requisites>>",
    #             course_content[course_code].co_requisites
    #              )
    # UniTec
    # Testing for pages
    # pages = Unitec_BSC_Prog_Descriptors()
    # count = 0
    # for apage in pages:
    #     print("Key? ",count,apage)
    #     count += 1
    # Testing for Courses
    #    course_content = Unitec_BSC_Prog_Descriptors()
    #    for key in course_content:
    #        print(key,':',course_content[key].full_name,':',course_content[key].aim)

# WinTec
#    course_content = Wintec_BAppliedIT_Vol2()
#    for key in course_content:
#        if not ("none" in course_content[key].co_requisites.lower() or "nil" in course_content[key].co_requisites.lower()) :
#             print(key,":",course_content[key].aim,"\n","     pre_requisite:",course_content[key].prerequisites,"\n","     co_requisite:",course_content[key].co_requisites)   

# UCol
# course_content = Ucol_Bachelor_of_Information_and_Communications_Technology_L7_Courses()
# for key in course_content:
#     #if not ("none" in course_content[key].co_requisites.lower() or "nil" in course_content[key].co_requisites.lower()) :
#          print(key,":",course_content[key].aim,"\n","     pre_requisite:",course_content[key].prequistes,"\n","     co_requisite:",course_content[key].co_requisites)

# WandW
# print(PR5006_HV4701_BIT())
# course_content = PR5006_HV4701_BIT()
# for key in course_content:
#       print(key,":",course_content[key].aim,"\n","     pre_requisite:",course_content[key].prequistes)
