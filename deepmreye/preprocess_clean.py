import os
import pickle
import numpy as np
import ants
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots


fn_base = '/mnt/CYHSM/Projects/DeepMReye/data/'
fn_rcollin27 = fn_base + 'rcollin27/rcolin27_t1_tal_lin.nii'
# Average functional mask! #fn_base + 'rcollin27/rcolin27_eyemask_big.nii'
fn_rcollin27_eye_mask = '/mnt/CYHSM/Projects/DeepMReye/data/rcollin27/combined_eye_mask.nii'

# --------------------------------------------------------------------------------
# --------------------------ANTS TRANSFORMS---------------------------------------
# --------------------------------------------------------------------------------
def transform_to_mni(fn_func, fn_rcollin27=fn_rcollin27, save_anat=True):
    # Read all images in ants format
    func = ants.image_read(fn_func)
    func_mean = func.get_average_of_timeseries()
    rcollin27 = ants.image_read(fn_rcollin27)

    # Use ants to registrate anat to collin27
    # anat_to_mni = ants.registration(fixed=rcollin27, moving=anat, type_of_transform='SyN', syn_metric='CC', syn_sampling=4)
    # V-2, fit directly to functional
    anat_to_mni = ants.registration(fixed=rcollin27, moving=func_mean, type_of_transform='SyNBold')

    # Transform (pre-registered) func to collin27 space
    func_mni = ants.apply_transforms(fixed=rcollin27, moving=func,
                                     transformlist=anat_to_mni['fwdtransforms'], imagetype=3)

    # Save functional image
    if save_anat:
        fn_func_mni = os.path.dirname(fn_func) + os.path.sep + 'collin27_' + os.path.basename(fn_func)
        ants.image_write(func_mni, fn_func_mni)
    return func_mni


def transform_to_mni_with_anatomical(fn_func, fn_anat, fn_rcollin27=fn_rcollin27, save_anat=True):
    # Read all images in ants format
    anat = ants.image_read(fn_anat)
    func = ants.image_read(fn_func)
    rcollin27 = ants.image_read(fn_rcollin27)

    # Use ants to registrate anat to collin27
    anat_to_mni = ants.registration(fixed=rcollin27, moving=anat,
                                    type_of_transform='SyN', syn_metric='CC', syn_sampling=4)

    # Transform (pre-registered) func to collin27 space
    func_mni = ants.apply_transforms(fixed=rcollin27, moving=func,
                                     transformlist=anat_to_mni['fwdtransforms'], imagetype=3)

    # Save functional image
    if save_anat:
        fn_func_mni = os.path.dirname(fn_func) + os.path.sep + 'collin27_' + os.path.basename(fn_func)
        ants.image_write(func_mni, fn_func_mni)
    return func_mni


def register_to_eye_masks(dme_template, func, masks, verbose=1, transforms=None, metric='GC'):
    """Register functional to DeepMReye template (dme_template) using different sized masks

    Parameters
    ----------
    dme_template : ants Image
        Ants image with template file
    func : ants Image
        Functional image to register to dme_template
    masks : list
        List of Ants image objects containing variable sized masks
    verbose : int, optional
        Verbosity level of function, by default 1
    transforms : string, optional
        Which transforms should be used to transform image, by default None & set to Similarity
    metric : str, optional
        Which metric to quantify fit, by default 'GC'

    Returns
    -------
    func : ants Image
        Functional image registered to dme_template
    transformation_stats : array
        Statistics of transformation, used for dataset report
    """
    transformation_stats = []
    for idx, mask in enumerate(masks):
        if transforms is None:
            type_of_transform = 'Similarity'
        else:
            type_of_transform = transforms[idx]
        register_to_nau = ants.registration(fixed=dme_template, moving=func.get_average_of_timeseries(), aff_random_sampling_rate=1,
                                            type_of_transform=type_of_transform, mask=mask, aff_metric=metric, aff_sampling=512,
                                            aff_iterations=(200, 200, 200, 10), aff_smoothing_sigmas=(0, 0, 0, 0))
        if verbose > 0:
            if 'SyN' in type_of_transform:
                registered_fwd = loadmat(register_to_nau['fwdtransforms'][1])['AffineTransform_float_3_3']
            else:
                registered_fwd = loadmat(register_to_nau['fwdtransforms'][0])['AffineTransform_float_3_3']
            print('Mask {}/{}, Sum: {}, Mean {}, Std {}, Median {}'.format(idx, len(masks)-1, np.sum(registered_fwd),
                                                                           np.mean(registered_fwd), np.std(registered_fwd), np.median(registered_fwd)))
        transformation_stats.append(np.mean(registered_fwd))
        # Transform
        func = ants.apply_transforms(fixed=dme_template, moving=func,
                                     transformlist=register_to_nau['fwdtransforms'], imagetype=3)
    return func, np.array(transformation_stats)

def run_participant(fp_func, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges, replace_with=0):
    """Run preprocessing for one participant with templates and masks preloaded to avoid computational overhead

    Parameters
    ----------
    fp_func : string
        Filepath to participant functional
    dme_template : ants Image
        Preloaded Image to dme_template
    eyemask_big : ants Image
        Big eyemask as ants Image
    eyemask_small : ants Image
        Small eyemask as ants Image
    x_edges : list
        Edges of mask in x-dimension
    y_edges : list
        Edges of mask in y-dimension
    z_edges : list
        Edges of mask in z-dimension
    replace_with : int, optional
        Values outside of mask are set to this, by default 0
    """
    # Load subject specific run. File should be Nifti and 4D but should also work with other formats which can be read with AntsPy
    func = ants.image_read(fp_func)
    # Register to deepmreye template (dme_template). If registration fails quality check, try below line with additional parameter "transforms=['Affine', 'Affine', 'SyNAggro']"
    transform_to_dme, transformation_statistics = register_to_eye_masks(dme_template, func, masks=[None, eyemask_big, eyemask_small])
    # Cut mask and save to subject folder with subject report / quality control plots
    cut_mask(transform_to_dme, eyemask_small.numpy(), x_edges, y_edges, z_edges, replace_with=replace_with, save_overview=True, fp_func=fp_func)

# --------------------------------------------------------------------------------
# --------------------------MASKING-----------------------------------------------
# --------------------------------------------------------------------------------

def get_mask_edges(fp_mask, split=True):
    """Gets edges of mask

    Parameters
    ----------
    fp_mask : filepath, optional
        Filepath to mask
    split : bool, optional
        Splits masks into hemispheres, by default True

    Returns
    -------
    mask:
        Array of extracted mask edges
    x_edges, y_edges, z_edges: 
        Edges in (x,y,z)-dimension
    """
    # Read to ants format
    mask = ants.image_read(fp_mask)
    # Get indices for left and right eye seperately
    edge_indices = np.where(mask.numpy() == 1)
    if split:
        # Get left and right based on middle between left and right eye. For collin27 : 45
        middle_cut = (edge_indices[0][np.argmax(np.diff(edge_indices[0]))] +
                      edge_indices[0][np.argmax(np.diff(edge_indices[0])) + 1]) // 2
        # Get x and y values for both eyes and combine in one volume
        left_indices = edge_indices[0][edge_indices[0] < middle_cut]
        right_indices = edge_indices[0][edge_indices[0] > middle_cut]
        x_edges = (np.max(left_indices), np.min(left_indices), np.max(right_indices), np.min(right_indices))
    else:
        middle = np.min(edge_indices[0]) + (np.max(edge_indices[0]) - np.min(edge_indices[0])) // 2
        x_edges = (middle, np.min(edge_indices[0]), np.max(edge_indices[0]), middle)
    y_edges = (np.max(edge_indices[1]), np.min(edge_indices[1]))
    z_edges = (np.max(edge_indices[2]), np.min(edge_indices[2]))

    return (mask.numpy(), x_edges, y_edges, z_edges)


def cut_mask(to_mask, mask, x_edges, y_edges, z_edges, replace_with=0, save_overview=True, fp_func=None, verbose=0):
    """Cut mask into given shape given edges

    Parameters
    ----------
    to_mask : ants Image
        Image to mask
    mask : ants Image
        Mask as numpy array
    x_edges : list
        Edges of mask in x-dimension
    y_edges : list
        Edges of mask in y-dimension
    z_edges : list
        Edges of mask in z-dimension
    replace_with : int, optional
        Values outside of mask are set to this, by default 0
    save_overview : bool, optional
        Saves report / quality control figure when set to True, by default True
    fp_func : str, optional
        Filepath to new functional, by default None
    verbose : int, optional
        Verbosity level of this function, by default 0

    Returns
    -------
    original_input : ants Image
        Returns to_mask
    masked_eye : ants Image
        masked_eye as numpy array
    mask : ants Image
        Return mask
    """
    # Mask image to set out of mask values
    original_input = to_mask.copy()
    to_mask[mask < 1, ...] = replace_with
    # Slice for mask
    masked_eye_left = to_mask[x_edges[1]: x_edges[0], y_edges[1]: y_edges[0], z_edges[1]: z_edges[0], ...]
    masked_eye_right = to_mask[x_edges[3]: x_edges[2], y_edges[1]: y_edges[0], z_edges[1]: z_edges[0], ...]
    masked_eye = np.concatenate((masked_eye_right, masked_eye_left))
    if verbose > 0:
        print('Voxels > 0 / Mean of voxels: {} / {}'.format(np.sum(np.mean(masked_eye, axis=3) > 0), np.mean(masked_eye)))
    # Save back masked func to .nii and masked eye to .p
    if save_overview:
        fn_full_mask = os.path.dirname(fp_func) + os.path.sep + 'mask_overview' + os.path.basename(fp_func)[:-4]
        plot_subject_report(fn_full_mask, original_input, masked_eye, mask)
    fn_masked_eye = os.path.dirname(fp_func) + os.path.sep + 'mask_' + os.path.basename(fp_func)[:-4] + '.p'
    pickle.dump(masked_eye, open(fn_masked_eye, 'wb'))
    
    return (original_input, masked_eye, mask)


def plot_overview(plot_values, save=True, fig_filename='./fig'):
    # Plot overview of three axes to check if eye is in mask
    plot_value_nozero = plot_values[plot_values > 1]
    fig, axes = plt.subplots(1, 5, figsize=(26, 10))
    im1 = axes[0].matshow(np.mean(plot_values, axis=0), aspect='auto', cmap='jet')
    im2 = axes[1].matshow(np.mean(plot_values, axis=1), aspect='auto', cmap='jet')
    im3 = axes[2].matshow(np.mean(plot_values, axis=2), aspect='auto', cmap='jet')
    images = [im1, im2, im3]
    axes[3].hist(plot_values.flatten(), bins=100)
    axes[3].grid(True)
    axes[4].hist(plot_value_nozero.flatten(), bins=100)
    axes[4].grid(True)
    for idx, ax in enumerate(axes[0:3]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(images[idx], cax=cax, orientation='vertical')
    if save:
        fig.savefig(fig_filename + '.png', dpi=100)
    else:
        fig.show()
    plt.close('all')


def plot_subject_report(fn_subject, original_input, masked_eye, mask, color="rgb(0, 150, 175)", bg_color="rgb(0,0,0)"):
    """Plots quality check figure for given subject

    Parameters
    ----------
    fn_subject : string
        Filepath to subject
    original_input : ants Image
        Filepath to functional image of subject
    masked_eye : array
        Numpy array of masked eye
    mask : ants Image
        ants mask
    color : str, optional
        Boxplot color, by default "rgb(0, 150, 175)"
    bg_color : str, optional
        Background color, by default "rgb(0,0,0)"
    """
    # Prepare data
    whole_brain_mask = original_input.get_average_of_timeseries()
    eye_mask = np.mean(masked_eye, axis=3)
    eye_mask_flat = eye_mask.flatten()
    # Also remove zero for histogram
    eye_mask_flat = eye_mask_flat[eye_mask_flat > 0]

    whole_brain_timecourse = np.mean(original_input.numpy(), axis=(0,1,2))
    eye_mask_timecourse = np.mean(masked_eye, axis=(0,1,2))

    # Normalize
    eye_mask_timecourse = (eye_mask_timecourse - np.min(eye_mask_timecourse)) / (np.max(eye_mask_timecourse) - np.min(eye_mask_timecourse))
    whole_brain_timecourse = (whole_brain_timecourse - np.min(whole_brain_timecourse)) / (np.max(whole_brain_timecourse) - np.min(whole_brain_timecourse))    
    
    # Plot
    fig = make_subplots(rows=2, cols=4, column_widths=[0.2, 0.2, 0.2, 0.4], row_heights=[0.6, 0.4], vertical_spacing=0.13)

    fig.add_trace(go.Heatmap(z=whole_brain_mask[25,:,:].transpose(), showscale=False, colorscale='Greys_r'),row=1, col=1)
    fig.add_trace(go.Heatmap(z=mask[25,:,:].transpose(), showscale=False, colorscale=[[0, 'rgba(0, 0, 0, 0)'], [1.0, 'rgba(255, 0, 0, 0.25)']]),row=1, col=1)
    fig.add_trace(go.Heatmap(z=whole_brain_mask[:,90,:].transpose(), showscale=False, colorscale='Greys_r'),row=1, col=2)
    fig.add_trace(go.Heatmap(z=mask[:,90,:].transpose(), showscale=False, colorscale=[[0, 'rgba(0, 0, 0, 0)'], [1.0, 'rgba(255, 0, 0, 0.25)']]),row=1, col=2)
    fig.add_trace(go.Heatmap(z=whole_brain_mask[:,:,15], showscale=False, colorscale='Greys_r'),row=1, col=3)
    fig.add_trace(go.Heatmap(z=mask[:,:,15], showscale=False, colorscale=[[0, 'rgba(0, 0, 0, 0)'], [1.0, 'rgba(255, 0, 0, 0.25)']]),row=1, col=3)

    fig.add_trace(go.Histogram(x=eye_mask_flat, nbinsx=75, marker = {'line' : {'width' : 0.75, 'color' : 'rgb(255, 255, 255)'}}, marker_color=color),row=1, col=4)

    fig.add_trace(go.Heatmap(z=np.mean(eye_mask, axis=0).transpose(), showscale=False, colorscale='Hot'),row=2, col=1)
    fig.add_trace(go.Heatmap(z=np.mean(eye_mask, axis=1).transpose(), showscale=False, colorscale='Hot'),row=2, col=2)
    fig.add_trace(go.Heatmap(z=np.mean(eye_mask, axis=2), showscale=False, colorscale='Hot'),row=2, col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, len(eye_mask_timecourse)), y=eye_mask_timecourse, marker_color=color),row=2, col=4)
    fig.add_trace(go.Scatter(x=np.arange(0, len(whole_brain_timecourse)), y=whole_brain_timecourse, marker_color='rgb(255, 255, 255)'),row=2, col=4)

    annotations = [dict(x=0.07, y=1.03, xref='paper', yref='paper', text="x=-20", font=(dict(size=20)), showarrow=False),
                  dict(x=0.29, y=1.03, xref='paper', yref='paper', text="y=36", font=(dict(size=20)), showarrow=False),
                  dict(x=0.54, y=1.03, xref='paper', yref='paper', text="z=-30", font=(dict(size=20)), showarrow=False),
                  dict(x=0.17, y=1.1, xref='paper', yref='paper', text="<b>Transformed MNI space with eye mask (r)</b>", font=(dict(size=20)), showarrow=False),
                  dict(x=0.93, y=1.1, xref='paper', yref='paper', text="<b>Histogram of eye mask voxels</b>", font=(dict(size=20)), showarrow=False),
                  dict(x=0.24, y=0.42, xref='paper', yref='paper', text="<b>Eye mask voxels</b>", font=(dict(size=20)), showarrow=False),
                  dict(x=0.99, y=0.41, xref='paper', yref='paper', text="<b>Average across whole brain (w) & eye mask (b)</b>", font=(dict(size=20)), showarrow=False)]

    fig.update_layout(autosize=False,showlegend=False, width=1400, height=600, margin=dict(t=70, l=30, b=50, r=30),
                          plot_bgcolor=bg_color, paper_bgcolor=bg_color, font = {'color' : '#FFFFFF', 'size' : 13}, annotations=annotations)
    fig.update_xaxes(showgrid=False, showticklabels=True, col=4)
    fig.update_yaxes(showgrid=False, showticklabels=True, col=4)
    fig.update_yaxes(showgrid=False, showticklabels=True, showline=True, col=4, row=1)
    # Remove labels from brain plots
    fig.update_yaxes(showgrid=False, showticklabels=False, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, col=3)
    fig.update_xaxes(showgrid=False, showticklabels=False, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, col=2)
    fig.update_xaxes(showgrid=False, showticklabels=False, col=3)
    # Add mean and median to hist
    fig.add_vline(x=np.mean(eye_mask_flat), annotation=dict(text='Mean', y=0.9), line=dict(color='rgb(255, 255, 255)'), row=1, col=4)
    fig.add_vline(x=np.median(eye_mask_flat), annotation=dict(text='Median'), line=dict(color='rgb(255, 255, 255)'), row=1, col=4)
    
    fn_full_mask = os.path.dirname(fn_subject) + os.path.sep + 'mask_report' + os.path.basename(fn_subject)[:-4]
    fig.write_html(fn_full_mask + ".html")
    
# --------------------------------------------------------------------------------
# -----------------------IMG MANIPULATIONS----------------------------------------
# --------------------------------------------------------------------------------


def normalize_img(img_in, mad_time=False, standardize_tr=True, std_cut_after=5):
    # Transpose so time comes first
    img_in = np.transpose(img_in, axes=(3, 0, 1, 2))
    zero_indices = img_in == 0
    img_in[zero_indices] = np.NaN

    # Median absolute deviation (MAD)
    if mad_time:
        est_mean = np.nanmedian(img_in, axis=0)
        est_std = np.nanmedian(abs(img_in - est_mean), axis=0)
        img_in = (img_in - est_mean) / est_std
    else:
        img_in = (img_in - np.nanmean(img_in, axis=0)) / np.nanstd(img_in, axis=0)

    # Normalize each functional on its own:
    if standardize_tr:
        img_in = np.array([(x - np.nanmean(x)) / np.nanstd(x) for x in img_in])
    if std_cut_after is not None:
        std_before = np.nanstd(img_in)
        img_in[img_in > std_cut_after*std_before] = std_cut_after*std_before
        img_in[img_in < -std_cut_after*std_before] = -std_cut_after*std_before

    # If division by zero replace with 0
    img_in[~np.isfinite(img_in)] = 0

    # Transpose back to original
    img_in[zero_indices] = 0
    img_in = np.transpose(img_in, axes=(1, 2, 3, 0))

    return img_in