from . import one_over_f
from jwst import datamodels
from jwst.stpipe import Step

__all__ = ["OneOverFStep"]


class OneOverFStep(Step):
    """
    OneOverFStep: Performs 1/f noise correction by computing a constant
    for each column from the input ramp data model.
    """

    class_alias = "oneoverf"

    spec = """
        outlier_map = str(default=None)
        iterative = boolean(default=False)
        save_intermediate = boolean(default=False)
        intermediate_output_subdir = str(default=None)
        mean_per_frame = boolean(default=False)
    """

    def process(self, input):
        with datamodels.RampModel(input) as input_model:

            # TODO: This raises warning about 0 when doing weight RMS calculation.
            result = one_over_f.correct_oof(
                input_model,
                output_dir=self.output_dir,
                outlier_map=self.outlier_map,
                iterative=self.iterative,
                save_intermediate=self.save_intermediate,
                intermediate_output_subdir=self.intermediate_output_subdir,
                mean_per_frame=self.mean_per_frame,
            )

            result.meta.cal_step.oneoverf = "COMPLETE"

        return result


if __name__ == "__main__":
    # Open the uncal time series that needs 1/f correction
    exposurename = (
        "../scratch/results/stage1/jw01189017001_06101_00001_nis_saturation.fits"
    )
    outdir = "scratch/results/oof_test"

    step = OneOverFStep()
    step.save_results = True
    step.run(exposurename)
