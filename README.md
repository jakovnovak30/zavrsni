# CppDiff - demonstracijski okvir za diferencijabilno programiranje

### O nama
Ovaj repozitorij je napravljen u sklopu završnog rada preddiplomskog studija računarstva na Fakultetu elektrotehnike i računarstva Sveučilišta u Zagrebu.
Autor rada je Jakov Novak, a mentor prof. dr. sc. Siniša Šegvić.
Sam rad možete preuzeti na [ovoj](./rad/rad.pdf) adresi.

### Ideja
Ideja je bila prikazati neuralne mreže kao matematičke funkcije te osmisliti jednostavnu platformu za automatsko diferenciranje istih kako bi se mogao olakšati dizajn i implementacija neuralnih mreža. Korišten je programski jezik C++ zajedno s OpenCL-om te clBlast-om za potrebne operacije s matricama.

### Trenutne funkcionalnosti
Za sad je moguće definirati proizvoljnu matematičku funkciju te istu evaluirati i nacrtati njezin graf te odrediti gradijent funkcije.
Osim tih opcija, moguće je optimirati funkcije preko SGD optimizatora te definirati razne oblike neuralnih mreža preko "Module" sučelja.
TODO: slika

### Budući rad
Dodat ću još nekoliko vrsta optimizatora, funkcija gubitka te eventalno opciju za učitavanje slika kako bi napravio demonstracijski klasifikator.
