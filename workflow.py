#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

from Pegasus.api import (
    Properties,
    Workflow,
    Transformation,
    TransformationCatalog,
    Site,
    SiteCatalog,
    ReplicaCatalog,
    Container,
    Arch,
    OS,
    Job,
    PegasusClientError,
)


class OcularoneBenchWorkflow:
    def __init__(self):
        """."""
        self.props = Properties()

        self.wf = Workflow("ocularone-bench-workflow")
        self.tc = TransformationCatalog()
        self.sc = SiteCatalog()
        self.rc = ReplicaCatalog()

        self.wf.add_transformation_catalog(self.tc)
        self.wf.add_site_catalog(self.sc)
        self.wf.add_replica_catalog(self.rc)

    def generate_props(self):
        self.props["pegasus.mode"] = "development"
        self.props["dagman.maxjobs"] = "3"
        self.props.write()

    def generate_tc(self):
        """."""
        container = Container(
            "ocularone-bench-workflow",
            Container.DOCKER,
            "docker://pegasus/ocularone-bench-workflow:latest",
        ).add_env(TORCH_HOME="/tmp/torch")
        self.tc.add_containers(container)

        pegasus_worker = Transformation(
            "worker",
            namespace="pegasus",
            pfn="https://download.pegasus.isi.edu/pegasus/5.0.9/pegasus-worker-5.0.9-aarch64_ubuntu_20.tar.gz",
            is_stageable=True,
            site="isi",
        )
        self.tc.add_transformations(pegasus_worker)

        base_path = Path("/usr/local/bin")
        for model in self.models:
            model_job = Transformation(
                f"model_{model.lower()}",
                site="local",
                pfn=base_path / "new_updated_inference.py",
                is_stageable=False,
                container=container,
            )
            self.tc.add_transformations(model_job)

        self.tc.add_transformations(
            Transformation(
                "post_model_hazard_vest",
                site="local",
                pfn=base_path / "hvt_detc_grpc.py",
                is_stageable=False,
                container=container,
            ),
            Transformation(
                "post_model_crowd_density",
                site="local",
                pfn=base_path / "crowd_detc_grpc.py",
                is_stageable=False,
                container=container,
            ),
            Transformation(
                "post_model_mask_detection",
                site="local",
                pfn=base_path / "mask_detc_grpc.py",
                is_stageable=False,
                container=container,
            ),
            Transformation(
                "post_model_distance_estimation_vip",
                site="local",
                pfn=base_path / "dist_est_vip_grpc.py",
                is_stageable=False,
                container=container,
            ),
            Transformation(
                "post_model_body_pose_estimation",
                site="local",
                pfn=base_path / "bp_est_grpc.py",
                is_stageable=False,
                container=container,
            ),
        )

    def generate_sc(self):
        """."""
        condorpool = (
            Site("condorpool", arch=Arch.X86_64, os_type=OS.LINUX)
            .add_pegasus_profile(style="condor")
            .add_condor_profile(universe="vanilla")
        )

        self.sc.add_sites(condorpool)

    def generate_workflow(self):
        """."""
        videos = {
            int(video.stem.split("-")[-1]): video for video in self.video_dir.iterdir()
        }
        first = True
        previous_job = {}
        previous_post_job = {}
        for task_id, video in sorted(videos.items()):
            task_id = f"task-{task_id}"

            self.rc.add_replica("local", video.name, str(video.resolve()))

            for model in self.models:
                model_output = f"{task_id}-{model.lower()}.txt"
                model_job = (
                    Job(f"model_{model.lower()}")
                    .add_args(
                        "--task-id",
                        task_id,
                        "--dnn-model",
                        model,
                        "--file",
                        video.name,
                        "--output",
                        model_output,
                    )
                    .add_inputs(video.name)
                    .add_outputs(model_output, register_replica=False)
                )

                self.wf.add_jobs(model_job)

                if first is False:
                    self.wf.add_dependency(previous_job[model], children=[model_job])

                previous_job[model] = model_job

                if model == "BODY_POSE_ESTIMATION":
                    continue

                model_post_output = f"{task_id}-{model.lower()}-post.txt"
                post_model_job = (
                    Job(f"post_model_{model.lower()}")
                    .add_args(
                        "--task-id",
                        task_id,
                        "--file",
                        model_output,
                    )
                    .add_inputs(model_output)
                    .set_stdout(model_post_output, register_replica=False)
                )

                self.wf.add_jobs(post_model_job)

                if first is False:
                    self.wf.add_dependency(
                        previous_post_job[model], children=[post_model_job]
                    )

                previous_post_job[model] = post_model_job

            first = False

    def plan_workflow(self):
        try:
            self.wf.plan(
                output_dir="output",
                sites=["condorpool"],
                dir="submit",
                verbose=5,
                cleanup="leaf",
                # submit=True,
            ).graph(include_files=True, label="xform-id", output="graph.png")
        except PegasusClientError as e:
            print(e.output)

    def __call__(self):
        """."""
        parser = argparse.ArgumentParser(description="generate a pegasus workflow")
        parser.add_argument(
            "--video-dir",
            dest="video_dir",
            default=None,
            required=True,
            help="Directory with mp4 files",
        )
        args = parser.parse_args(sys.argv[1:])

        self.video_dir = Path(args.video_dir).resolve()
        self.models = (
            "HAZARD_VEST",
            "CROWD_DENSITY",
            "MASK_DETECTION",
            "DISTANCE_ESTIMATION_VIP",
            "BODY_POSE_ESTIMATION",
        )

        if not self.video_dir.exists():
            raise ValueError("--video-dir either does not exist or is not readable")

        self.generate_props()
        self.generate_tc()
        self.generate_sc()
        self.generate_workflow()
        self.plan_workflow()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    wf = OcularoneBenchWorkflow()
    wf()
