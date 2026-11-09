"""Close every execution resource and retain all shutdown failures."""


def close_execution_resources(report, hands, cameras, client, arm):
    errors = []
    report['hand_shutdown'] = {}
    report['camera_shutdown'] = {}

    def close_one(resource, label):
        try:
            resource.close()
        except BaseException as error:
            detail = f'{label}: {type(error).__name__}: {error}'
            errors.append(detail)
            return {'status': 'error', 'error': detail}
        return {'status': 'closed'}

    for side, hand in hands.items():
        report['hand_shutdown'][side] = close_one(hand, f'hand/{side}')
    for role, camera in cameras.items():
        report['camera_shutdown'][role] = close_one(camera, f'camera/{role}')
    if client is not None:
        report['transport_shutdown'] = close_one(client, 'transport')
    report['arm_connection_shutdown'] = close_one(arm, 'arm_connection')
    if 'shutdown_error' in report:
        errors.insert(0, report['shutdown_error'])
    if errors:
        report['execution_status'] = report['status']
        report['status'] = 'aborted'
        report['shutdown_errors'] = errors
        report['shutdown_error'] = '; '.join(errors)
