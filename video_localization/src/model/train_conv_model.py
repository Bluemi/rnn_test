from data.data import Dataset
from model.conv_model import create_compiled_conv_model
from util.show_frames import show_frames, ZoomAnnotationsRenderer


def train_conv_model(args):
    train_dataset = Dataset.load_database(args.train_data)
    eval_dataset = None
    if args.eval_data is not None:
        eval_dataset = Dataset.load_database(args.eval_data)

    model = create_compiled_conv_model(train_dataset.get_resolution())
    model.fit(
        x=train_dataset.video_data,
        y=train_dataset.annotation_data,
        validation_data=(eval_dataset.video_data, eval_dataset.annotation_data) if eval_dataset else None,
        batch_size=30,
        epochs=40,
        verbose=True
    )

    if args.show:
        show_dataset = eval_dataset or train_dataset

        annotations = model.predict(x=show_dataset.video_data)

        show_frames(
            show_dataset,
            render_callback=ZoomAnnotationsRenderer(
                [show_dataset.annotation_data, annotations],
                train_dataset.get_resolution()
            )
        )
