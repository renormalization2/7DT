# %%
# ============================================================
# 	Library
# ------------------------------------------------------------
from ccdproc import ImageFileCollection

# ------------------------------------------------------------
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time

# time.sleep(60*60*2)
from datetime import datetime

# ------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time

# ------------------------------------------------------------
from preprocess import calib

# from util import tool

# ------------------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
# ------------------------------------------------------------
# Plot preset
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 500
plt.rc("font", family="serif")


# ------------------------------------------------------------
testflag = True
# ------------------------------------------------------------


# ------------------------------------------------------------
# 	Functions
# ------------------------------------------------------------
def calc_mean_dateloc(dateloclist):

    # 문자열을 datetime 객체로 변환
    datetime_objects = [datetime.fromisoformat(t) for t in dateloclist]

    # datetime 객체를 POSIX 시간으로 변환
    posix_times = [dt.timestamp() for dt in datetime_objects]

    # 평균 POSIX 시간 계산
    mean_posix_time = np.mean(posix_times)

    # 평균 POSIX 시간을 datetime 객체로 변환
    mean_datetime = datetime.fromtimestamp(mean_posix_time)

    # 필요한 경우, datetime 객체를 ISOT 형식의 문자열로 변환
    mean_isot_time = mean_datetime.isoformat()
    return mean_isot_time


def inputlist_gen():
    """
    A makeshift function for testing.
    To be deleted in a real pipeline
    """
    from glob import glob
    from pathlib import Path

    datadir = Path("/data3/dhhyun/7DTStackCode/mytest/data/NGC0253/7DT01/m650")
    filelist = glob(str(datadir / "calib*0.fits"))
    with open(datadir / "inputlist.txt", "w") as f:
        f.write("file\n")
        for filename in filelist:
            f.write(filename + "\n")


def ascii_parser(imagelist_file_to_stack):
    if os.path.exists(imagelist_file_to_stack):
        print(f"{imagelist_file_to_stack} found!")
    else:
        print(f"Not Found {imagelist_file_to_stack}!")
        sys.exit()
    input_table = Table.read(imagelist_file_to_stack, format="ascii")
    _files = [f for f in input_table["file"].data]
    return _files


def unpack(packed, type, ex=None):
    """returns an element of a collection. e.g., [a] -> a"""
    if len(packed) != 1:
        print(f"There are more than one ({len(packed)}) {type}s")
        unpacked = input(
            f"Type {type.upper()} name (e.g. {packed if ex is None else ex}):"
        )
    else:
        unpacked = packed[0]
    return unpacked


def mkdir(path, subdir=None):
    path = Path(path)
    if subdir is not None:
        path = path / subdir
    if not path.exists():
        path.mkdir(parents=True)  # exist_ok=False
    return str(path)


def safe_input(exstr):
    while True:
        # to be deleted
        if testflag:
            path_save_parent = Path("/data3/dhhyun/7DTStackCode/mytest/out")
            break

        path_save_parent = Path(input(exstr))
        if path_save_parent.exists():
            break
        else:
            print("Specified directory does not exist")
            print(f"Do you wish to make {path_save_parent.resolve()}? [Y] or n")
            perm = input("[Y] or n: ").lower()
            if (perm == "y") or (perm == "yes") or (perm == "[y]"):
                path_save_parent = Path(mkdir(path_save_parent))
                break
            else:
                print("Type path again.")
    return path_save_parent


def deg2hms(angle_deg):
    from astropy.coordinates import Angle
    import astropy.units as u

    angle = Angle(float(angle_deg) * u.deg)
    sign = "-" if angle < 0 else ""
    angle = abs(angle)
    hms = angle.hms
    return f"{sign}{int(hms.h):0>2}:{int(hms.m):0>2}:{int(hms.s):0>2}"


class SwarpCom:
    def run(self):
        # Setting Keywords
        # - the code should run without it but it allows more control
        # self.update_keys()

        # also you can do
        # self.zpkey = 'ZP'

        # sets miscellaneous variables
        ic = self.set_imcollection()

        # background subtraction
        # bkgval = self.blankfield
        self.bkgsub()
        # self.bkgsub(bkgval=bkgval)

        # zero point scaling
        self.zpscale()

        # swarp imcombine
        self.swarp_imcom()

    def __init__(self, imagelist_file_to_stack=None) -> None:
        # ------------------------------------------------------------
        # 	Path
        # ------------------------------------------------------------
        self.path_config = "./config"

        if imagelist_file_to_stack is None:
            imagelist_file_to_stack = input(f"Image List to Stack (/data/data.txt):")

        # not used?
        # self.path_calib = '/large_data/processed'

        # ------------------------------------------------------------
        # 	Setting
        # ------------------------------------------------------------

        # Unused?
        # self.keys = [
        #     "imagetyp",
        #     "telescop",
        #     "object",
        #     "filter",
        #     "exptime",
        #     "ul5_1",
        #     "seeing",
        #     "elong",
        #     "ellip",
        # ]

        # ------------------------------------------------------------
        # 	Keywords
        # ------------------------------------------------------------
        # 	Gain
        # gain_default = 0.779809474945068
        # 	ZeroPoint Key
        self.zpkey = f"ZP_AUTO"

        # 	Universal Facility Name
        self.obs = "7DT"

        self.keys_to_remember = [
            "EGAIN",
            "TELESCOP",
            "EGAIN",
            "FILTER",
            "OBJECT",
            "OBJCTRA",
            "OBJCTDEC",
            "JD",
            "MJD",
            "SKYVAL",
            "EXPTIME",
            self.zpkey,
        ]

        self.keywords_to_add = [
            "IMAGETYP",
            # "EXPOSURE",
            # "EXPTIME",
            "DATE-LOC",
            # "DATE-OBS",
            "XBINNING",
            "YBINNING",
            # "GAIN",
            "EGAIN",
            "XPIXSZ",
            "YPIXSZ",
            "INSTRUME",
            "SET-TEMP",
            "CCD-TEMP",
            "TELESCOP",
            "FOCALLEN",
            "FOCRATIO",
            "RA",
            "DEC",
            "CENTALT",
            "CENTAZ",
            "AIRMASS",
            "PIERSIDE",
            "SITEELEV",
            "SITELAT",
            "SITELONG",
            "FWHEEL",
            "FILTER",
            "OBJECT",
            "OBJCTRA",
            "OBJCTDEC",
            "OBJCTROT",
            "FOCNAME",
            "FOCPOS",
            "FOCUSPOS",
            "FOCUSSZ",
            "ROWORDER",
            # "COMMENT",
            "_QUINOX",
            "SWCREATE",
        ]

        self._files = ascii_parser(imagelist_file_to_stack)

        if testflag:
            self._files = self._files[:3]

    def update_keys(self, new_keys):
        self.keys_to_remember = new_keys

    def update_keywords(self, new_keywords):
        self.keywords_to_add = new_keywords

    def set_imcollection(self):
        import re

        # ------------------------------------------------------------
        # 	Setting Metadata
        # ------------------------------------------------------------
        print(f"Reading images... (takes a few mins)")
        # 	Get Image Collection (takes some time)
        ic = ImageFileCollection(filenames=self._files, keywords=self.keys_to_remember)
        self.files = ic.files
        self.n_stack = len(self.files)
        self.zpvalues = ic.summary[self.zpkey].data
        self.skyvalues = ic.summary["SKYVAL"].data
        self.objra = ic.summary["OBJCTRA"].data[0].replace(" ", ":")
        self.objdec = ic.summary["OBJCTDEC"].data[0].replace(" ", ":")
        self.mjd_stacked = np.mean(ic.summary["MJD"].data)
        # 	Total Exposure Time [sec]
        self.total_exptime = np.sum(ic.summary["EXPTIME"])

        objs = np.unique(ic.summary["OBJECT"].data)
        filters = np.unique(ic.summary["FILTER"].data)
        egains = np.unique(ic.summary["EGAIN"].data)
        print(f"OBJECT(s): {objs} (N={len(objs)})")
        print(f"FILTER(s): {filters} (N={len(filters)})")
        print(f"EGAIN(s): {egains} (N={len(egains)})")
        self.obj = unpack(objs, "object")
        self.filte = unpack(filters, "filter", ex="m650")
        self.gain_default = unpack(egains, "egain", ex="0.256190478801727")

        #   Get center coord. from predefined tiles
        if bool(re.search("T\d{5}", self.obj)):
            from astropy.coordinates import Angle
            import astropy.units as u

            center = ascii.read("/data4/gecko/GECKO/S240422ed/data/displaycenter.txt")
            objrow = center[int(self.obj[1:])]
            self.objra = deg2hms(objrow["ra"])
            self.objdec = deg2hms(objrow["dec"])

        # ------------------------------------------------------------
        # 	Base Image for Image Alignment
        # ------------------------------------------------------------
        self.baseim = self.files[0]
        self.zp_base = ic.summary[self.zpkey][0]

        # ------------------------------------------------------------
        # 	Print Input Summary
        # ------------------------------------------------------------
        print(f"Input Images to Stack ({len(self.files):_}):")
        for ii, inim in enumerate(self.files):
            print(f"[{ii:>6}] {os.path.basename(inim)}")
            if ii > 10:
                print("...")
                break

        # ------------------------------------------------------------
        #   Define Paths
        # ------------------------------------------------------------
        exstr = f"Path to save results (/large_data/Commission): "
        path_save_parent = safe_input(exstr)

        path_save = path_save_parent / self.obj / self.filte
        print("Save Directory:", path_save.resolve())

        self.path_save = mkdir(path_save)
        self.path_imagelist = Path(path_save) / "images_to_stack.txt"
        self.path_bkgsub = mkdir(path_save, "bkgsub")
        self.path_scaled = mkdir(path_save, "scaled")
        self.path_resamp = mkdir(path_save, "resamp")

        return ic

    def bkgsub(self):
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        print("BACKGROUND Subtraction...")
        _st = time.time()
        bkg_subtracted_images = []
        for ii, (inim, _bkg) in enumerate(zip(self.files, self.skyvalues)):
            print(f"[{ii:>6}] {os.path.basename(inim)}", end="")
            nim = f"{self.path_bkgsub}/{os.path.basename(inim).replace('fits', 'bkgsub.fits')}"
            if not os.path.exists(nim):
                with fits.open(inim, memmap=True) as hdul:  # 파일 열기
                    _data = hdul[0].data  # 데이터 접근
                    _hdr = hdul[0].header  # 헤더 접근
                    # _bkg = np.median(_data)
                    _data -= _bkg
                    print(f"- {_bkg:.3f}")
                    fits.writeto(nim, _data, header=_hdr, overwrite=True)
            bkg_subtracted_images.append(nim)
        self.bkg_subtracted_images = bkg_subtracted_images
        _delt = time.time() - _st
        print(f"--> Done ({_delt:.1f}sec)")

    def sift(self, filenames):
        zplist = [fits.getheader(f)[self.zpkey] for f in filenames]
        mask = [True for zp in zplist if not np.isnan(zp) else False]
        return np.array(filenames)[mask]

    def zpscale(self):
        bkg_subtracted_images = self.bkg_subtracted_images
        zpvalues = self.zpvalues

        # ------------------------------------------------------------
        # 	ZP Scale
        # ------------------------------------------------------------
        print(f"Flux Scale to ZP={self.zp_base}")
        zpscaled_images = []
        _st = time.time()
        for ii, (inim, _zp) in enumerate(zip(bkg_subtracted_images, zpvalues)):
            print(f"[{ii:>6}] {os.path.basename(inim)}", end=" ")
            _fscaled_image = f"{self.path_scaled}/{os.path.basename(inim).replace('fits', 'zpscaled.fits')}"
            if not os.path.exists(_fscaled_image):
                with fits.open(inim, memmap=True) as hdul:  # 파일 열기
                    _data = hdul[0].data  # 데이터 접근
                    _hdr = hdul[0].header  # 헤더 접근
                    _fscale = 10 ** (0.4 * (self.zp_base - _zp))
                    _fscaled_data = _data * _fscale
                    print(
                        f"x {_fscale:.3f}",
                    )
                    fits.writeto(_fscaled_image, _fscaled_data, _hdr, overwrite=True)
            zpscaled_images.append(_fscaled_image)
        self.zpscaled_images = zpscaled_images
        _delt = time.time() - _st
        print(f"--> Done ({_delt:.1f}sec)")


    def swarp_imcom(self):
        # ------------------------------------------------------------
        # 	Images to Combine for SWarp
        # ------------------------------------------------------------
        with open(self.path_imagelist, "w") as f:
            for inim in self.zpscaled_images:
                f.write(f"{inim}\n")

        # 	Get Header info
        exptime_stacked = self.total_exptime
        mjd_stacked = self.mjd_stacked
        jd_stacked = Time(mjd_stacked, format="mjd").jd
        dateobs_stacked = Time(mjd_stacked, format="mjd").isot
        # airmass_stacked = np.mean(airmasslist)
        # dateloc_stacked = calc_mean_dateloc(dateloclist)
        # alt_stacked = np.mean(altlist)
        # az_stacked = np.mean(azlist)

        center = f"{self.objra},{self.objdec}"
        datestr, timestr = calib.extract_date_and_time(dateobs_stacked)
        comim = f"{self.path_save}/calib_{self.obs}_{self.obj}_{datestr}_{timestr}_{self.filte}_{exptime_stacked:g}.com.fits"
        weightim = comim.replace("com", "weight")

        # ------------------------------------------------------------
        # 	Image Combine
        # ------------------------------------------------------------
        t0_stack = time.time()
        print(f"Total Exptime: {self.total_exptime}")
        gain = (2 / 3) * self.n_stack * self.gain_default
        # 	SWarp
        # swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} -GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} -FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
        # swarpcom = f"swarp -c {path_config}/7dt.nocom.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center}"
        swarpcom = (
            f"swarp -c {self.path_config}/7dt.swarp @{self.path_imagelist} "
            f"-IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} "
            f"-SUBTRACT_BACK N -RESAMPLE_DIR {self.path_resamp} "
            f"-GAIN_KEYWORD EGAIN -GAIN_DEFAULT {self.gain_default} "
            f"-FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
        )

        print(swarpcom)
        os.system(swarpcom)

        # 	Get Genenral Header from Base Image
        with fits.open(self.baseim) as hdulist:
            header = hdulist[0].header
            chdr = {key: header.get(key, None) for key in self.keywords_to_add}

        # 	Put General Header Infomation on the Combined Image
        with fits.open(comim) as hdulist:
            data = hdulist[0].data
            header = hdulist[0].header
            for key in list(chdr.keys()):
                header[key] = chdr[key]

        # 	Additional Header Information
        keywords_to_update = {
            "DATE-OBS": (
                dateobs_stacked,
                "Time of observation (UTC) for combined image",
            ),
            # 'DATE-LOC': (dateloc_stacked, 'Time of observation (local) for combined image'),
            "EXPTIME": (
                exptime_stacked,
                "[s] Total exposure duration for combined image",
            ),
            "EXPOSURE": (
                exptime_stacked,
                "[s] Total exposure duration for combined image",
            ),
            # 'CENTALT' : (alt_stacked,     '[deg] Average altitude of telescope for combined image'),
            # 'CENTAZ'  : (az_stacked,      '[deg] Average azimuth of telescope for combined image'),
            # 'AIRMASS' : (airmass_stacked, 'Average airmass at frame center for combined image (Gueymard 1993)'),
            "MJD": (
                mjd_stacked,
                "Modified Julian Date at start of observations for combined image",
            ),
            "JD": (
                jd_stacked,
                "Julian Date at start of observations for combined image",
            ),
            "SKYVAL": (0, "SKY MEDIAN VALUE (Subtracted)"),
            "GAIN": (gain, "Sensor gain"),
        }

        # 	Header Update
        with fits.open(comim, mode="update") as hdul:
            # 헤더 정보 가져오기
            header = hdul[0].header

            # 여러 헤더 항목 업데이트
            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Stacked Single images
            # for nn, inim in enumerate(files):
            # 	header[f"IMG{nn:0>5}"] = (os.path.basename(inim), "")

            # 변경 사항 저장
            hdul.flush()

        delt_stack = time.time() - t0_stack

        print(f"Time to stack {self.n_stack} images: {delt_stack:.3f} sec")


# %% Example
if __name__ == "__main__":

    imagelist_file_to_stack = (
        "/data3/dhhyun/7DTStackCode/mytest/data/NGC0253/7DT01/m650/inputlist.txt"
    )

    # if the path is not given, you will be prompted to give one
    SwarpCom(imagelist_file_to_stack).run()
    # imcom = SwarpCom(imagelist_file_to_stack)
    # imcom.run()
