import { staticFile } from 'remotion';
import { Composition } from 'remotion';
import { AudiogramComposition } from './Composition';
import './style.css';

const fps = 30;
const durationInFrames = 214 * fps;

export const RemotionRoot: React.FC = () => {
	return (
		<>
			<Composition
				id="Audiogram"
				component={AudiogramComposition}
				durationInFrames={durationInFrames}
				fps={fps}
				width={1080}
				height={1920}
				defaultProps={{
					audioOffsetInFrames: Math.round(0.5 * fps),
					source: staticFile('subtitles.srt'),
				}}
			/>
		</>
	);
};
