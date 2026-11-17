import { expect, test, vi } from 'vitest';
import { BrowserApi } from '../src/api';
import { createFollowCam } from '../src/camera';

test('the follow cam says so when WebGL is missing', async () => {
  // Node has no WebGL, so the renderer cannot start here.
  const container = { textContent: '', append: () => {} } as unknown as HTMLElement;
  createFollowCam(new BrowserApi(), () => null).mount(container);
  await vi.waitFor(() => expect(container.textContent).toMatch(/WebGL/));
});
