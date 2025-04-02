## TF-IDF Retriever

In this notebook, we will develop a TF-IDF Retriever on a small dataset. Let's import the TFIDFRetriever class


```python
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.TF_IDFRetriever import TFIDFRetriever
```


```python
# Create TF-IDF retriever
retriever = TFIDFRetriever()
```


```python
# Let us load the BNS sections
def load_md_files(base_folder):
    md_files_dict = []

    # Iterate through all folders in the base directory
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through all .md files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".md"):
                    file_path = os.path.join(folder_path, filename)

                    # Open the file and read its contents
                    with open(file_path, "r", encoding="utf-8") as file:
                        file_contents = file.read()

                    # Store the contents in the dictionary with the key being "folder/filename"
                    temp_doc = {}
                    temp_doc["_id"] = f"{folder}/{filename}"
                    temp_doc["text"] = file_contents
                    md_files_dict.append(temp_doc)

    return md_files_dict
```


```python
bns_data = load_md_files("ilab_sdg/")
```


```python
# Add some documents
for each_section in bns_data:
    retriever.add_document(each_section["_id"], each_section["text"])

retriever.update_index()
```


```python
# Search for a query
print("Search for 'robbery':")
results = retriever.search("robbery")
print(results)

# Get the matching documents
print("\nTop matching documents:")
for doc_id, score in results:
    print(f"Document {doc_id} (Score: {score:.4f}): {retriever.documents[doc_id]}")
```

    Search for 'robbery':
    [('Chapter_XVII/Section_311.md', 0.3283260006317209), ('Chapter_XVII/Section_312.md', 0.3116363975419594), ('Chapter_XVII/Section_313.md', 0.25107645287498526), ('Chapter_IV/Section_56.md', 0.2252681703552038), ('Chapter_III/Section_35.md', 0.2065372123740925), ('Chapter_XIV/Section_254.md', 0.20380948750630068), ('Chapter_XVII/Section_310.md', 0.18244384710317216), ('Chapter_XVII/Section_309.md', 0.14342161585986243), ('Chapter_IV/Section_59.md', 0.1432756833333348), ('Chapter_III/Section_41.md', 0.11395446924549021)]
    
    Top matching documents:
    Document Chapter_XVII/Section_311.md (Score: 0.3283): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 311: Robbery, or dacoity, with attempt to cause death or grievous hurt
    If, at the time of committing robbery or dacoity, the offender uses any deadly weapon, or causes grievous hurt to any person, or attempts to cause death or grievous hurt to any person, the imprisonment with which such offender shall be punished shall not be less than seven years.
    
    
    Document Chapter_XVII/Section_312.md (Score: 0.3116): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 312: Attempt to commit robbery or dacoity when armed with deadly weapon
    If, at the time of attempting to commit robbery or dacoity, the offender is armed with any deadly weapon, the imprisonment with which such offender shall be punished shall not be less than seven years.
    
    
    
    Document Chapter_XVII/Section_313.md (Score: 0.2511): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 313: Punishment for belonging to gang of robbers, etc.
    Whoever belongs to any gang of persons associated in habitually committing theft or robbery, and not being a gang of dacoits, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine.
    
    
    
    Document Chapter_IV/Section_56.md (Score: 0.2253): CHAPTER IV: OF ABETMENT, CRIMINAL CONSPIRACY AND ATTEMPT
    
    Subchapter: of abetment
    
    Section 56: Abetment of offence punishable with imprisonment.
    Whoever abets an offence punishable with imprisonment shall, if that offence be not committed in consequence of the abetment, and no express provision is made under this Sanhita for the punishment of such abetment, be punished with imprisonment of any description provided for that offence for a term which may extend to one-fourth part of the longest term provided for that offence; or with such fine as is provided for that offence, or with both; and if the abettor or the person abetted is a public servant, whose duty it is to prevent the commission of such offence, the abettor shall be punished with imprisonment of any description provided for that offence, for a term which may extend to one-half of the longest term provided for that offence, or with such fine as is provided for the offence, or with both.
    Illustrations.
    (a) A instigates B to give false evidence. Here, if B does not give false evidence, A has nevertheless committed the offence defined in this section, and is punishable accordingly.
    (b) A, a police officer, whose duty it is to prevent robbery, abets the commission of robbery. Here, though the robbery be not committed, A is liable to one-half of the longest term of imprisonment provided for that offence, and also to fine. (c) B abets the commission of a robbery by A, a police officer, whose duty it is to prevent that offence. Here, though the robbery be not committed, B is liable to one-half of the longest term of imprisonment provided for the offence of robbery, and also to fine.
    
    
    Document Chapter_III/Section_35.md (Score: 0.2065): CHAPTER III: GENERAL EXCEPTIONS
    
    Subchapter: Of right of private defence
    
    Section 35: Right of private defence of body and of property
    Every person has a right, subject to the restrictions contained in section 37, to defend (a) his own body, and the body of any other person, against any offence affecting the human body; (b) the property, whether movable or immovable, of himself or of any other person, against any act which is an offence falling under the definition of theft, robbery, mischief or criminal trespass, or which is an attempt to commit theft, robbery, mischief or criminal trespass.
    
    
    Document Chapter_XIV/Section_254.md (Score: 0.2038): CHAPTER XIV: OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE
    
    Section 254: Penalty for harbouring robbers or dacoits
    Whoever, knowing or having reason to believe that any persons are about to commit or have recently committed robbery or dacoity, harbours them or any of them, with the intention of facilitating the commission of such robbery or dacoity, or of screening them or any of them from punishment, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine.
    Explanation: For the purposes of this section it is immaterial whether the robbery or dacoity is intended to be committed, or has been committed, within or without India.
    Exception: The provisions of this section do not extend to the case in which the harbour is by the spouse of the offender.
    
    
    Document Chapter_XVII/Section_310.md (Score: 0.1824): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 310: Dacoity
    (1) When five or more persons conjointly commit or attempt to commit a robbery, or where the whole number of persons conjointly committing or attempting to commit a robbery, and persons present and aiding such commission or attempt, amount to five or more, every person so committing, attempting or aiding, is said to commit dacoity. (2) Whoever commits dacoity shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine. (3) If any one of five or more persons, who are conjointly committing dacoity, commits murder in so committing dacoity, every one of those persons shall be punished with death, or imprisonment for life, or rigorous imprisonment for a term which shall not be less than ten years, and shall also be liable to fine. (4) Whoever makes any preparation for committing dacoity, shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine. (5) Whoever is one of five or more persons assembled for the purpose of committing dacoity, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine. (6) Whoever belongs to a gang of persons associated for the purpose of habitually committing dacoity, shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.
    
    
    Document Chapter_XVII/Section_309.md (Score: 0.1434): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 309: Robbery
    (1) In all robbery there is either theft or extortion. (2) Theft is robbery if, in order to the committing of the theft, or in committing the theft, or in carrying away or attempting to carry away property obtained by the theft, the offender, for that end voluntarily causes or attempts to cause to any person death or hurt or wrongful restraint, or fear of instant death or of instant hurt, or of instant wrongful restraint. (3) Extortion is robbery if the offender, at the time of committing the extortion, is in the presence of the person put in fear, and commits the extortion by putting that person in fear of instant death, of instant hurt, or of instant wrongful restraint to that person or to some other person, and, by so putting in fear, induces the person so put in fear then and there to deliver up the thing extorted.
    Explanation: The offender is said to be present if he is sufficiently near to put the other person in fear of instant death, of instant hurt, or of instant wrongful restraint.
    Illustrations.
    (a) A holds Z down, and fraudulently takes Zs money and jewels from Zs clothes, without Zs consent. Here A has committed theft, and, in order to the committing of that theft, has voluntarily caused wrongful restraint to Z. A has therefore committed robbery.
    (b) A meets Z on the high road, shows a pistol, and demands Zs purse. Z, in consequence, surrenders his purse. Here A has extorted the purse from Z by putting him in fear of instant hurt, and being at the time of committing the extortion in his presence. A has therefore committed robbery. (c) A meets Z and Zs child on the high road. A takes the child, and threatens to fling it down a precipice, unless Z delivers his purse. Z, in consequence, delivers his purse. Here A has extorted the purse from Z, by causing Z to be in fear of instant hurt to the child who is there present. A has therefore committed robbery on Z. (d) A obtains property from Z by sayingYour child is in the hands of my gang, and will be put to death unless you send us ten thousand rupees. This is extortion, and punishable as such; but it is not robbery, unless Z is put in fear of the instant death of his child. (4) Whoever commits robbery shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine; and, if the robbery be committed on the highway between sunset and sunrise, the imprisonment may be extended to fourteen years. (5) Whoever attempts to commit robbery shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine. (6) If any person, in committing or in attempting to commit robbery, voluntarily causes hurt, such person, and any other person jointly concerned in committing or attempting to commit such robbery, shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.
    
    
    Document Chapter_IV/Section_59.md (Score: 0.1433): CHAPTER IV: OF ABETMENT, CRIMINAL CONSPIRACY AND ATTEMPT
    
    Subchapter: of abetment
    
    Section 59: Public servant concealing design to commit offence which it is his duty to prevent
    Whoever, being a public servant, intending to facilitate or knowing it to be likely that he will thereby facilitate the commission of an offence which it is his duty as such public servant to prevent, voluntarily conceals, by any act or omission or by the use of encryption or any other information hiding tool, the existence of a design to commit such offence, or makes any representation which he knows to be false respecting such design shall, (a) if the offence be committed, be punished with imprisonment of any description provided for the offence, for a term which may extend to one-half of the longest term of such imprisonment, or with such fine as is provided for that offence, or with both; or (b) if the offence be punishable with death or imprisonment for life, with imprisonment of either description for a term which may extend to ten years; or (c) if the offence be not committed, shall be punished with imprisonment of any description provided for the offence for a term which may extend to one-fourth part of the longest term of such imprisonment or with such fine as is provided for the offence, or with both.
    Illustration.
    A, an officer of police, being legally bound to give information of all designs to commit robbery which may come to his knowledge, and knowing that B designs to commit robbery, omits to give such information, with intent to so facilitate the commission of that offence.
    Here A has by an illegal omission concealed the existence of Bs design, and is liable to punishment according to the provision of this section.
    
    
    Document Chapter_III/Section_41.md (Score: 0.1140): CHAPTER III: GENERAL EXCEPTIONS
    
    Subchapter: Of right of private defence
    
    Section 41: When right of private defence of property extends to causing death
    The right of private defence of property extends, under the restrictions specified in section 37, to the voluntary causing of death or of any other harm to the wrong-doer, if the offence, the committing of which, or the attempting to commit which, occasions the exercise of the right, be an offence of any of the descriptions hereinafter enumerated, namely: (a) robbery; (b) house-breaking after sunset and before sunrise; (c) mischief by fire or any explosive substance committed on any building, tent or vessel, which building, tent or vessel is used as a human dwelling, or as a place for the custody of property; (d) theft, mischief, or house-trespass, under such circumstances as may reasonably cause apprehension that death or grievous hurt will be the consequence, if such right of private defence is not exercised.
    
    


    



```python
# Search for a query
print("Search for 'robbery and chain-snatching':")
results = retriever.search("robbery and chain-snatching")
print(results)

# Get the matching documents
print("\nTop matching documents:")
for doc_id, score in results:
    print(f"Document {doc_id} (Score: {score:.4f}): {retriever.documents[doc_id]}")
```

    Search for 'robbery and chain-snatching':
    [('Chapter_XVII/Section_304.md', 0.33812669799394435), ('Chapter_XVII/Section_311.md', 0.17157725031788007), ('Chapter_XVII/Section_312.md', 0.16285556454968503), ('Chapter_XVII/Section_313.md', 0.1313873776686545), ('Chapter_VI/Section_112.md', 0.13036309055536316), ('Chapter_IV/Section_56.md', 0.11783448296494729), ('Chapter_III/Section_35.md', 0.10803659298491969), ('Chapter_XIV/Section_254.md', 0.10657189517963878), ('Chapter_XVII/Section_310.md', 0.09551570002679212), ('Chapter_XVII/Section_309.md', 0.07503556081766766)]
    
    Top matching documents:
    Document Chapter_XVII/Section_304.md (Score: 0.3381): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of theft
    
    Section 304: Snatching
    (1) Theft is snatching if, in order to commit theft, the offender suddenly or quickly or forcibly seizes or secures or grabs or takes away from any person or from his possession any movable property. (2) Whoever commits snatching, shall be punished with imprisonment of either description for a term which may extend to three years, and shall also be liable to fine.
    
    
    Document Chapter_XVII/Section_311.md (Score: 0.1716): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 311: Robbery, or dacoity, with attempt to cause death or grievous hurt
    If, at the time of committing robbery or dacoity, the offender uses any deadly weapon, or causes grievous hurt to any person, or attempts to cause death or grievous hurt to any person, the imprisonment with which such offender shall be punished shall not be less than seven years.
    
    
    Document Chapter_XVII/Section_312.md (Score: 0.1629): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 312: Attempt to commit robbery or dacoity when armed with deadly weapon
    If, at the time of attempting to commit robbery or dacoity, the offender is armed with any deadly weapon, the imprisonment with which such offender shall be punished shall not be less than seven years.
    
    
    
    Document Chapter_XVII/Section_313.md (Score: 0.1314): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 313: Punishment for belonging to gang of robbers, etc.
    Whoever belongs to any gang of persons associated in habitually committing theft or robbery, and not being a gang of dacoits, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine.
    
    
    
    Document Chapter_VI/Section_112.md (Score: 0.1304): CHAPTER VI: OF OFFENCES AFFECTING THE HUMAN BODY
    
    Subchapter: Of offences affecting life
    
    Section 112: Petty organised crime
    (1) Whoever, being a member of a group or gang, either singly or jointly, commits any act of theft, snatching, cheating, unauthorised selling of tickets, unauthorised betting or gambling, selling of public examination question papers or any other similar criminal act, is said to commit petty organised crime.
    Explanation: For the purposes of this sub-section "theft" includes trick theft, theft from vehicle, dwelling house or business premises, cargo theft, pick pocketing, theft through card skimming, shoplifting and theft of Automated Teller Machine. (2) Whoever commits any petty organised crime shall be punished with imprisonment for a term which shall not be less than one year but which may extend to seven years, and shall also be liable to fine.
    
    
    
    Document Chapter_IV/Section_56.md (Score: 0.1178): CHAPTER IV: OF ABETMENT, CRIMINAL CONSPIRACY AND ATTEMPT
    
    Subchapter: of abetment
    
    Section 56: Abetment of offence punishable with imprisonment.
    Whoever abets an offence punishable with imprisonment shall, if that offence be not committed in consequence of the abetment, and no express provision is made under this Sanhita for the punishment of such abetment, be punished with imprisonment of any description provided for that offence for a term which may extend to one-fourth part of the longest term provided for that offence; or with such fine as is provided for that offence, or with both; and if the abettor or the person abetted is a public servant, whose duty it is to prevent the commission of such offence, the abettor shall be punished with imprisonment of any description provided for that offence, for a term which may extend to one-half of the longest term provided for that offence, or with such fine as is provided for the offence, or with both.
    Illustrations.
    (a) A instigates B to give false evidence. Here, if B does not give false evidence, A has nevertheless committed the offence defined in this section, and is punishable accordingly.
    (b) A, a police officer, whose duty it is to prevent robbery, abets the commission of robbery. Here, though the robbery be not committed, A is liable to one-half of the longest term of imprisonment provided for that offence, and also to fine. (c) B abets the commission of a robbery by A, a police officer, whose duty it is to prevent that offence. Here, though the robbery be not committed, B is liable to one-half of the longest term of imprisonment provided for the offence of robbery, and also to fine.
    
    
    Document Chapter_III/Section_35.md (Score: 0.1080): CHAPTER III: GENERAL EXCEPTIONS
    
    Subchapter: Of right of private defence
    
    Section 35: Right of private defence of body and of property
    Every person has a right, subject to the restrictions contained in section 37, to defend (a) his own body, and the body of any other person, against any offence affecting the human body; (b) the property, whether movable or immovable, of himself or of any other person, against any act which is an offence falling under the definition of theft, robbery, mischief or criminal trespass, or which is an attempt to commit theft, robbery, mischief or criminal trespass.
    
    
    Document Chapter_XIV/Section_254.md (Score: 0.1066): CHAPTER XIV: OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE
    
    Section 254: Penalty for harbouring robbers or dacoits
    Whoever, knowing or having reason to believe that any persons are about to commit or have recently committed robbery or dacoity, harbours them or any of them, with the intention of facilitating the commission of such robbery or dacoity, or of screening them or any of them from punishment, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine.
    Explanation: For the purposes of this section it is immaterial whether the robbery or dacoity is intended to be committed, or has been committed, within or without India.
    Exception: The provisions of this section do not extend to the case in which the harbour is by the spouse of the offender.
    
    
    Document Chapter_XVII/Section_310.md (Score: 0.0955): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 310: Dacoity
    (1) When five or more persons conjointly commit or attempt to commit a robbery, or where the whole number of persons conjointly committing or attempting to commit a robbery, and persons present and aiding such commission or attempt, amount to five or more, every person so committing, attempting or aiding, is said to commit dacoity. (2) Whoever commits dacoity shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine. (3) If any one of five or more persons, who are conjointly committing dacoity, commits murder in so committing dacoity, every one of those persons shall be punished with death, or imprisonment for life, or rigorous imprisonment for a term which shall not be less than ten years, and shall also be liable to fine. (4) Whoever makes any preparation for committing dacoity, shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine. (5) Whoever is one of five or more persons assembled for the purpose of committing dacoity, shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine. (6) Whoever belongs to a gang of persons associated for the purpose of habitually committing dacoity, shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.
    
    
    Document Chapter_XVII/Section_309.md (Score: 0.0750): CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    
    Subchapter: Of robbery and dacoity
    
    Section 309: Robbery
    (1) In all robbery there is either theft or extortion. (2) Theft is robbery if, in order to the committing of the theft, or in committing the theft, or in carrying away or attempting to carry away property obtained by the theft, the offender, for that end voluntarily causes or attempts to cause to any person death or hurt or wrongful restraint, or fear of instant death or of instant hurt, or of instant wrongful restraint. (3) Extortion is robbery if the offender, at the time of committing the extortion, is in the presence of the person put in fear, and commits the extortion by putting that person in fear of instant death, of instant hurt, or of instant wrongful restraint to that person or to some other person, and, by so putting in fear, induces the person so put in fear then and there to deliver up the thing extorted.
    Explanation: The offender is said to be present if he is sufficiently near to put the other person in fear of instant death, of instant hurt, or of instant wrongful restraint.
    Illustrations.
    (a) A holds Z down, and fraudulently takes Zs money and jewels from Zs clothes, without Zs consent. Here A has committed theft, and, in order to the committing of that theft, has voluntarily caused wrongful restraint to Z. A has therefore committed robbery.
    (b) A meets Z on the high road, shows a pistol, and demands Zs purse. Z, in consequence, surrenders his purse. Here A has extorted the purse from Z by putting him in fear of instant hurt, and being at the time of committing the extortion in his presence. A has therefore committed robbery. (c) A meets Z and Zs child on the high road. A takes the child, and threatens to fling it down a precipice, unless Z delivers his purse. Z, in consequence, delivers his purse. Here A has extorted the purse from Z, by causing Z to be in fear of instant hurt to the child who is there present. A has therefore committed robbery on Z. (d) A obtains property from Z by sayingYour child is in the hands of my gang, and will be put to death unless you send us ten thousand rupees. This is extortion, and punishable as such; but it is not robbery, unless Z is put in fear of the instant death of his child. (4) Whoever commits robbery shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine; and, if the robbery be committed on the highway between sunset and sunrise, the imprisonment may be extended to fourteen years. (5) Whoever attempts to commit robbery shall be punished with rigorous imprisonment for a term which may extend to seven years, and shall also be liable to fine. (6) If any person, in committing or in attempting to commit robbery, voluntarily causes hurt, such person, and any other person jointly concerned in committing or attempting to commit such robbery, shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.
    
    



```python

```
