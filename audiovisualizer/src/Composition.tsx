import { useAudioData, visualizeAudio } from '@remotion/media-utils';
import React, { useEffect, useRef, useState } from 'react';
import {
	AbsoluteFill,
	Audio,
	continueRender,
	delayRender,
	Img,
	Sequence,
	useCurrentFrame,
	useVideoConfig,
	random,
} from 'remotion';
import audioSource from './assets/lilac.mp3';
import coverImg from './assets/brunomars.png';
import pauseImg from './assets/pause.png';
import fastForwardImg from './assets/fastForward.png';
import volumeImg from './assets/volume.png';
import { LINE_HEIGHT, PaginatedSubtitles } from './Subtitles';

interface AudioProps {
	type: 'front' | 'back';
}

function AudioViz(props: AudioProps) {
	const frame = useCurrentFrame();
	const { fps } = useVideoConfig();
	const audioData = useAudioData(audioSource);
	const { type } = props;

	if (!audioData) {
		return null;
	}

	const allVisualizationValues = visualizeAudio({
		fps,
		frame,
		audioData,
		numberOfSamples: 128, // Use more samples to get a nicer visualisation
	});

	// Pick the low values because they look nicer than high values
	// feel free to play around :)
	const visualization = allVisualizationValues.slice(7, 7 + 20 * 2);

	const mirrored = [...visualization.slice(1).reverse(), ...visualization];

	return (
		<div
			className={type === 'front' ? 'audio-viz1' : 'audio-viz2'}
			style={{
				justifySelf: 'center',
			}}
		>
			{mirrored.map((v, i) => {
				return (
					i % 2 === 0 && (
						<div
							key={i}
							className={type === 'front' ? 'bar1' : 'bar2'}
							style={{
								height: `${
									type === 'front'
										? random(null) * 100 < 70
											? 300 * v ** 0.6 + 2
											: 230 * v ** 0.6 + 2
										: random(null) * 100 < 70
										? 230 * v ** 0.6 + 2
										: 180 * v ** 0.6 + 2
								}%`,
							}}
						/>
					)
				);
			})}
		</div>
	);
}

export const AudiogramComposition: React.FC<{
	source: string;
	audioOffsetInFrames: number;
}> = ({ source, audioOffsetInFrames }) => {
	const ref = useRef<HTMLDivElement>(null);

	return (
		<div ref={ref}>
			<AbsoluteFill>
				<Sequence from={-audioOffsetInFrames}>
					<Audio src={audioSource} />

					<div
						className="container"
						style={{
							fontFamily: 'IBM Plex Sans',
							justifyContent: 'center',
						}}
					>
						<div className="bgImage" />
						<div
							style={{
								position: 'absolute',
								top: 'auto',
								bottom: 'auto',
								left: '-5%',
								zIndex: -1,
								height: '2000px',
								width: '110%',
								filter: 'blur(8px)',
							}}
						>
							<AudioViz type="back" />
						</div>
						<div
							className="musicContainer"
							style={{
								width: '85%',
								height: '80%',
								borderRadius: '90px',
								marginLeft: 'auto',
								marginRight: 'auto',
							}}
						>
							<div
								style={{
									height: '90%',
									width: '90%',
									marginLeft: 'auto',
									marginRight: 'auto',
									marginTop: '65px',
								}}
							>
								<Img
									src={coverImg}
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
										borderRadius: '30px',
										width: '100%',
									}}
								/>
								<div
									className="title1"
									style={{
										marginTop: '30px',
									}}
								>
									IU(아이유) - LILAC(라일락)
								</div>
								<div className="title2">Bruno Mars (A.I. cover) Base Model</div>
								<div style={{ height: '200px' }}>
									<AudioViz type="front" />
								</div>
								<div
									style={{
										display: 'flex',
										flex: 'column',
										justifyContent: 'center',
									}}
								>
									<Img
										src={fastForwardImg}
										style={{
											marginTop: 'auto',
											marginBottom: 'auto',
											height: '70px',
											transform: 'scaleX(-1)',
											position: 'relative',
										}}
									/>
									<Img
										src={pauseImg}
										style={{
											height: '120px',
											marginRight: '100px',
											marginLeft: '100px',
										}}
									/>
									<Img
										src={fastForwardImg}
										style={{
											marginTop: 'auto',
											marginBottom: 'auto',
											height: '70px',
											position: 'relative',
										}}
									/>
								</div>
								<Img
									src={volumeImg}
									style={{
										marginTop: 'auto',
										marginBottom: 'auto',
										width: '100%',
										position: 'relative',
									}}
								/>
							</div>
						</div>
					</div>
				</Sequence>
			</AbsoluteFill>
		</div>
	);
};
