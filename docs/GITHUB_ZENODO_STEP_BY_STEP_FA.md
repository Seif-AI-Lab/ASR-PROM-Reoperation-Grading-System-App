# راهنمای مرحله‌به‌مرحله آپلود در GitHub و گرفتن DOI از Zenodo

## 1) ساخت repository جدید در GitHub

1. وارد GitHub شوید.
2. بالا سمت راست روی علامت `+` کلیک کنید.
3. گزینه `New repository` را بزنید.
4. در قسمت `Repository name` بنویسید:

```text
asr-prom-reoperation-risk
```

5. در قسمت `Description` دقیقاً این متن را بنویسید:

```text
Code for ASR PROM-based lumbar reoperation risk prediction, dynamic ODI analysis, and risk grading.
```

6. فعلاً `Private` را انتخاب کنید.
7. گزینه `Add a README file` را فعال نکنید، چون فایل README داخل همین بسته آماده شده است.
8. گزینه `.gitignore` را هم انتخاب نکنید، چون فایل آماده داخل همین بسته وجود دارد.
9. گزینه `Choose a license` را هم انتخاب نکنید، چون فایل LICENSE داخل همین بسته وجود دارد.
10. روی `Create repository` کلیک کنید.

## 2) آپلود فایل‌های این بسته

1. فایل ZIP را روی کامپیوتر خود باز کنید.
2. وارد فولدر `asr-prom-reoperation-risk` شوید.
3. داخل صفحه repository در GitHub روی `Add file` کلیک کنید.
4. گزینه `Upload files` را بزنید.
5. همه فایل‌ها و فولدرهای داخل `asr-prom-reoperation-risk` را drag and drop کنید.
6. پایین صفحه در قسمت commit message بنویسید:

```text
Initial analysis code release
```

7. روی `Commit changes` کلیک کنید.

## 3) چک کردن بعد از upload

بعد از upload باید این ساختار را ببینید:

```text
README.md
requirements.txt
LICENSE
CITATION.cff
.gitignore
notebooks/
scripts/
docs/
data/
outputs/
```

داخل `notebooks/` باید این دو فایل باشد:

```text
Step_1_Analysis_and_Grading_System.ipynb
Step_2_Analysis.ipynb
```

## 4) وقتی مقاله آماده شد repository را Public کنید

1. وارد repository شوید.
2. بروید به `Settings`.
3. پایین صفحه بخش `Danger Zone`.
4. گزینه تغییر visibility را بزنید.
5. repository را از `Private` به `Public` تغییر دهید.

## 5) ساخت Release در GitHub

1. در صفحه اصلی repository، سمت راست یا بالا روی `Releases` کلیک کنید.
2. روی `Create a new release` یا `Draft a new release` کلیک کنید.
3. در قسمت tag بنویسید:

```text
v1.0.0
```

4. در قسمت release title بنویسید:

```text
Version 1.0.0
```

5. در قسمت description بنویسید:

```text
Initial archived release of the analysis code.
```

6. روی `Publish release` کلیک کنید.

## 6) وصل کردن GitHub به Zenodo

1. وارد Zenodo شوید.
2. با حساب GitHub وارد شوید یا حساب Zenodo را به GitHub وصل کنید.
3. در Zenodo وارد بخش `GitHub` شوید.
4. روی `Sync now` کلیک کنید.
5. repository با نام `asr-prom-reoperation-risk` را پیدا کنید.
6. دکمه کنار آن را روشن کنید تا repository به Zenodo وصل شود.

## 7) گرفتن DOI

1. بعد از اینکه repository در Zenodo فعال شد، در GitHub دوباره یک release بسازید یا همان release `v1.0.0` را publish کنید.
2. Zenodo release را archive می‌کند.
3. در صفحه Zenodo record، DOI را کپی کنید.
4. DOI معمولاً شبیه این است:

```text
https://doi.org/10.5281/zenodo.XXXXXXX
```

## 8) آپدیت نهایی repository بعد از DOI

بعد از گرفتن DOI:

1. فایل `README.md` را در GitHub باز کنید.
2. جای این متن را عوض کنید:

```text
https://doi.org/10.5281/zenodo.XXXXXXX
```

3. DOI واقعی Zenodo را جایگزین کنید.
4. اگر خواستید، در `CITATION.cff` هم DOI را اضافه کنید.

## 9) متن کوتاه برای بخش Code Availability

```text
The analysis code is available in a public GitHub repository and archived on Zenodo with a DOI. The original registry data are not publicly available because of data-use restrictions and patient privacy requirements.
```
