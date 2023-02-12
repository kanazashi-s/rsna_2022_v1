import os
os.environ['CUDA_MODULE_LOADING']='LAZY'

import sys
import shutil
from typing import List, cast, Iterable
import pandas as pd
import numpy as np
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import pydicom
from pydicom.valuerep import VR

sys.path.append("/tmp/nvjpeg2k-python/build/")
import dicomsdl
import nvjpeg2k
from cfg.general import GeneralCFG


def apply_voi_lut(
    arr: "np.ndarray",
    ds: "Dataset",
    index: int = 0,
    prefer_lut: bool = True
) -> "np.ndarray":
    """Apply a VOI lookup table or windowing operation to `arr`.

    .. versionadded:: 1.4

    .. versionchanged:: 2.1

        Added the `prefer_lut` keyword parameter

    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the VOI LUT or windowing operation
        to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`VOI LUT Module<part03/sect_C.11.2.html>`.
        If (0028,3010) *VOI LUT Sequence* is present then returns an array
        of ``np.uint8`` or ``np.uint16``, depending on the 3rd value of
        (0028,3002) *LUT Descriptor*. If (0028,1050) *Window Center* and
        (0028,1051) *Window Width* are present then returns an array of
        ``np.float64``. If neither are present then `arr` will be returned
        unchanged.
    index : int, optional
        When the VOI LUT Module contains multiple alternative views, this is
        the index of the view to return (default ``0``).
    prefer_lut : bool
        When the VOI LUT Module contains both *Window Width*/*Window Center*
        and *VOI LUT Sequence*, if ``True`` (default) then apply the VOI LUT,
        otherwise apply the windowing operation.

    Returns
    -------
    numpy.ndarray
        An array with applied VOI LUT or windowing operation.

    Notes
    -----
    When the dataset requires a modality LUT or rescale operation as part of
    the Modality LUT module then that must be applied before any windowing
    operation.

    See Also
    --------
    :func:`~pydicom.pixel_data_handlers.util.apply_modality_lut`
    :func:`~pydicom.pixel_data_handlers.util.apply_voi`
    :func:`~pydicom.pixel_data_handlers.util.apply_windowing`

    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.2
      <part03/sect_C.11.html#sect_C.11.2>`
    * DICOM Standard, Part 3, :dcm:`Annex C.8.11.3.1.5
      <part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    valid_voi = False
    if ds.get('VOILUTSequence'):
        ds.VOILUTSequence = cast(List["Dataset"], ds.VOILUTSequence)
        valid_voi = None not in [
            ds.VOILUTSequence[0].get('LUTDescriptor', None),
            ds.VOILUTSequence[0].get('LUTData', None)
        ]
    valid_windowing = None not in [
        ds.get('WindowCenter', None),
        ds.get('WindowWidth', None)
    ]

    if valid_voi and valid_windowing:
        if prefer_lut:
            return apply_voi(arr, ds, index)

        return apply_windowing(arr, ds, index)

    if valid_voi:
        return apply_voi(arr, ds, index)

    if valid_windowing:
        return apply_windowing(arr, ds, index)

    return arr


def apply_voi(
    arr: "np.ndarray", ds: "Dataset", index: int = 0
) -> "np.ndarray":
    """Apply a VOI lookup table to `arr`.

    .. versionadded:: 2.1

    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the VOI LUT to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`VOI LUT Module<part03/sect_C.11.2.html>`.
        If (0028,3010) *VOI LUT Sequence* is present then returns an array
        of ``np.uint8`` or ``np.uint16``, depending on the 3rd value of
        (0028,3002) *LUT Descriptor*, otherwise `arr` will be returned
        unchanged.
    index : int, optional
        When the VOI LUT Module contains multiple alternative views, this is
        the index of the view to return (default ``0``).

    Returns
    -------
    numpy.ndarray
        An array with applied VOI LUT.

    See Also
    --------
    :func:`~pydicom.pixel_data_handlers.util.apply_modality_lut`
    :func:`~pydicom.pixel_data_handlers.util.apply_windowing`

    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.2
      <part03/sect_C.11.html#sect_C.11.2>`
    * DICOM Standard, Part 3, :dcm:`Annex C.8.11.3.1.5
      <part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    if not ds.get('VOILUTSequence'):
        return arr

    if not np.issubdtype(arr.dtype, np.integer):
        print(#warnings.warn
            "Applying a VOI LUT on a float input array may give "
            "incorrect results"
        )

    # VOI LUT Sequence contains one or more items
    item = cast(List["Dataset"], ds.VOILUTSequence)[index]
    lut_descriptor = cast(List[int], item.LUTDescriptor)
    nr_entries = lut_descriptor[0] or 2**16
    first_map = lut_descriptor[1]

    # PS3.3 C.8.11.3.1.5: may be 8, 10-16
    nominal_depth = lut_descriptor[2]
    if nominal_depth in list(range(10, 17)):
        dtype = 'uint16'
    elif nominal_depth == 8:
        dtype = 'uint8'
    else:
        raise NotImplementedError(
            f"'{nominal_depth}' bits per LUT entry is not supported"
        )

    # Ambiguous VR, US or OW
    unc_data: Iterable[int]
    if item['LUTData'].VR == VR.OW:
        endianness = '<' if ds.is_little_endian else '>'
        unpack_fmt = f'{endianness}{nr_entries}H'
        unc_data = unpack(unpack_fmt, cast(bytes, item.LUTData))
    else:
        unc_data = cast(List[int], item.LUTData)

    lut_data: "np.ndarray" = np.asarray(unc_data, dtype=dtype)

    # IVs < `first_map` get set to first LUT entry (i.e. index 0)
    clipped_iv = np.zeros(arr.shape, dtype=dtype)
    # IVs >= `first_map` are mapped by the VOI LUT
    # `first_map` may be negative, positive or 0
    mapped_pixels = arr >= first_map
    clipped_iv[mapped_pixels] = arr[mapped_pixels] - first_map
    # IVs > number of entries get set to last entry
    np.clip(clipped_iv, 0, nr_entries - 1, out=clipped_iv)

    return cast("np.ndarray", lut_data[clipped_iv])


def apply_windowing(
    arr: "np.ndarray", ds: "Dataset", index: int = 0
) -> "np.ndarray":
    """Apply a windowing operation to `arr`.

    .. versionadded:: 2.1

    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the windowing operation to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`VOI LUT Module<part03/sect_C.11.2.html>`.
        If (0028,1050) *Window Center* and (0028,1051) *Window Width* are
        present then returns an array of ``np.float64``, otherwise `arr` will
        be returned unchanged.
    index : int, optional
        When the VOI LUT Module contains multiple alternative views, this is
        the index of the view to return (default ``0``).

    Returns
    -------
    numpy.ndarray
        An array with applied windowing operation.

    Notes
    -----
    When the dataset requires a modality LUT or rescale operation as part of
    the Modality LUT module then that must be applied before any windowing
    operation.

    See Also
    --------
    :func:`~pydicom.pixel_data_handlers.util.apply_modality_lut`
    :func:`~pydicom.pixel_data_handlers.util.apply_voi`

    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.2
      <part03/sect_C.11.html#sect_C.11.2>`
    * DICOM Standard, Part 3, :dcm:`Annex C.8.11.3.1.5
      <part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    if "WindowWidth" not in ds and "WindowCenter" not in ds:
        return arr

    if ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise ValueError(
            "When performing a windowing operation only 'MONOCHROME1' and "
            "'MONOCHROME2' are allowed for (0028,0004) Photometric "
            "Interpretation"
        )

    # May be LINEAR (default), LINEAR_EXACT, SIGMOID or not present, VM 1
    voi_func = cast(str, getattr(ds, 'VOILUTFunction', 'LINEAR')).upper()
    # VR DS, VM 1-n
    elem = ds['WindowCenter']
    center = (
        cast(List[float], elem.value)[index] if elem.VM > 1 else elem.value
    )
    center = cast(float, center)
    elem = ds['WindowWidth']
    width = cast(List[float], elem.value)[index] if elem.VM > 1 else elem.value
    width = cast(float, width)

    # The output range depends on whether or not a modality LUT or rescale
    #   operation has been applied
    ds.BitsStored = cast(int, ds.BitsStored)
    y_min: float
    y_max: float
    if ds.get('ModalityLUTSequence'):
        # Unsigned - see PS3.3 C.11.1.1.1
        y_min = 0
        item = cast(List["Dataset"], ds.ModalityLUTSequence)[0]
        bit_depth = cast(List[int], item.LUTDescriptor)[2]
        y_max = 2**bit_depth - 1
    elif ds.PixelRepresentation == 0:
        # Unsigned
        y_min = 0
        y_max = 2**ds.BitsStored - 1
    else:
        # Signed
        y_min = -2**(ds.BitsStored - 1)
        y_max = 2**(ds.BitsStored - 1) - 1

    slope = ds.get('RescaleSlope', None)
    intercept = ds.get('RescaleIntercept', None)
    if slope is not None and intercept is not None:
        ds.RescaleSlope = cast(float, ds.RescaleSlope)
        ds.RescaleIntercept = cast(float, ds.RescaleIntercept)
        # Otherwise its the actual data range
        y_min = y_min * ds.RescaleSlope + ds.RescaleIntercept
        y_max = y_max * ds.RescaleSlope + ds.RescaleIntercept

    y_range = y_max - y_min
    arr = arr.astype('float32')
    #arr = arr.astype('float64')

    if voi_func in ['LINEAR', 'LINEAR_EXACT']:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation"
                )
            center -= 0.5
            width -= 1
        elif width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'LINEAR_EXACT' windowing operation"
            )

        below = arr <= (center - width / 2)
        above = arr > (center + width / 2)
        between = np.logical_and(~below, ~above)

        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = (
                ((arr[between] - center) / width + 0.5) * y_range + y_min
            )
    elif voi_func == 'SIGMOID':
        # PS3.3 C.11.2.1.3.1
        if width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'SIGMOID' windowing operation"
            )

        arr = y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'"
        )

    return arr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t)/60
        hour = t // 60
        minutes = t % 60
        return '%2d hr %02d min' % (hour, minutes)

    elif mode=='sec':
        t = int(t)
        minutes = t // 60
        seconds = t % 60
        return '%2d min %02d sec' % (minutes, seconds)

    else:
        raise NotImplementedError


def read_image(df, image_dir):
    image = []
    for t, d in df.iterrows():
        image_file = f'{image_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png'
        m = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
        image.append(m)
    return image


def make_transfer_syntax_uid(df, dcm_dir):
    machine_id_to_transfer = {}
    machine_id = df.machine_id.unique()
    for i in machine_id:
        d = df[df.machine_id == i].iloc[0]
        f = f'{dcm_dir}/{d.patient_id}/{d.image_id}.dcm'
        dicom = pydicom.dcmread(f)
        machine_id_to_transfer[i] = dicom.file_meta.TransferSyntaxUID
    return machine_id_to_transfer


def normalised_to_8bit(image, photometric_interpretation):
    xmin = image.min()
    xmax = image.max()

    norm = np.empty_like(image, dtype=np.uint8)
    dicomsdl.util.convert_to_uint8(image, norm, xmin, xmax)
    if photometric_interpretation == 'MONOCHROME1':
        norm = 255 - norm
    return norm


def normalised_to_16bit(image, photometric_interpretation):
    xmin = image.min()
    xmax = image.max()
    assert xmin >= 0 and xmax <= 65535
    assert xmax > 1  # 0 - 1 画像が入力されたりしないか？

    image_uint16 = image.astype(np.uint16)
    if photometric_interpretation == 'MONOCHROME1':
        image_uint16 = 65535 - image_uint16
    return image_uint16


def resize_image_to_height(image, image_height, photometric_interpretation):
    h, w = image.shape[:2]
    s = image_height/h
    if image_height != h:
        image = cv2.resize(image, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    new_image_width = image.shape[1]
    is_horizontal_flip = image[:, :new_image_width//2].sum() < image[:, new_image_width//2:].sum()
    if photometric_interpretation == 'MONOCHROME1':
        is_horizontal_flip = not is_horizontal_flip
    if is_horizontal_flip:
        image = np.fliplr(image)

    padding_dict = {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 1410 - new_image_width,
    }
    image = cv2.copyMakeBorder(image, **padding_dict, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image

# ----------------------------------------------------------------
# dicomsdl reader


def dicomsdl_to_numpy_image(ds, index=0):
    # https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    info = ds.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')

    shape = [info['Rows'], info['Cols']]
    dtype = info['dtype']
    outarr = np.empty(shape, dtype=dtype)
    ds.copyFrameData(index, outarr)
    return outarr


def dicomsdl_parallel_process(d, dcm_dir, image_dir, image_height, is_voi_lut):
    dcm_file = f'{dcm_dir}/{d.patient_id}/{d.image_id}.dcm'
    ds = dicomsdl.open(dcm_file)
    image = dicomsdl_to_numpy_image(ds)
    image = resize_image_to_height(image, image_height, ds.PhotometricInterpretation)

    if is_voi_lut:
        dc = pydicom.dcmread(dcm_file)
        image = apply_voi_lut(image, dc)
        image = image.astype(np.float32)
    image = normalised_to_8bit(image, ds.PhotometricInterpretation)  # +1

    # save as png
    os.makedirs(f'{image_dir}/{d.machine_id}/{d.patient_id}', exist_ok=True)
    cv2.imwrite(f'{image_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png', image)


def process_non_j2k(df, dcm_dir, image_dir, image_height, n_jobs, is_voi_lut=True):
    # https://stackoverflow.com/questions/56659294/does-joblib-parallel-keep-the-original-order-of-data-passed
    # Parallel(n_jobs=2, backend='multiprocessing')(

    Parallel(n_jobs=n_jobs)(
        delayed(dicomsdl_parallel_process)(d, dcm_dir, image_dir, image_height, is_voi_lut)
        for t, d in tqdm(df.iterrows())
    )

# ----------------------------------------------------------------
# nvjpeg2k reader


'''
TransferSyntaxUID
1.2.840.10008.1.2.4.70 = JPEG Lossless, Nonhierarchical, First- Order Prediction (Processes 14)
1.2.840.10008.1.2.4.90 = JPEG 2000 Image Compression (Lossless Only)
'''
j2k_decoder = nvjpeg2k.Decoder()


def process_j2k(df, dcm_dir, image_dir, image_height, is_voi_lut=True):
    for t, d in tqdm(df.iterrows()):
        dcm_file = f'{dcm_dir}/{d.patient_id}/{d.image_id}.dcm'
        dc = pydicom.dcmread(dcm_file)
        offset = dc.PixelData.find(b'\x00\x00\x00\x0C')
        jpeg_stream = bytearray(dc.PixelData[offset:])
        image = j2k_decoder.decode(jpeg_stream)
        image = resize_image_to_height(image, image_height, dc.PhotometricInterpretation)

        if is_voi_lut:
            image = apply_voi_lut(image, dc)
            image = image.astype(np.float32)
        image = normalised_to_8bit(image, dc.PhotometricInterpretation)
        # image = normalised_to_16bit(image, dc.PhotometricInterpretation)

        # save as png
        os.makedirs(f'{image_dir}/{d.machine_id}/{d.patient_id}', exist_ok=True)
        cv2.imwrite(f'{image_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png', image)


def main():
    convert_height = 1536

    csv_file = GeneralCFG.raw_data_dir / "test.csv"
    dcm_dir = GeneralCFG.raw_data_dir / "test_images"
    png_dir = GeneralCFG.png_converted_dir / f"1536_ker_png_test"

    shutil.rmtree(png_dir, ignore_errors=True)

    train_df = pd.read_csv(csv_file)
    machine_id_to_transfer = make_transfer_syntax_uid(train_df, dcm_dir)
    train_df.loc[:, 'i'] = np.arange(len(train_df))
    train_df.loc[:, 'TransferSyntaxUID'] = train_df.machine_id.map(machine_id_to_transfer)

    j2k_df = train_df[train_df.TransferSyntaxUID == '1.2.840.10008.1.2.4.90'].reset_index(drop=True)
    non_j2k_df = train_df[train_df.TransferSyntaxUID != '1.2.840.10008.1.2.4.90'].reset_index(drop=True)

    print(f'process_j2k(): {len(j2k_df)}')
    process_j2k(j2k_df, dcm_dir, png_dir, convert_height)

    print(f'process_non_j2k(): {len(non_j2k_df)}')
    process_non_j2k(non_j2k_df, dcm_dir, png_dir, convert_height, n_jobs=2)


if __name__ == "__main__":
    main()
